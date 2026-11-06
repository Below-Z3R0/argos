// Package qdrant is the only vector store Argos talks to. It is not hidden
// behind a vendor interface on purpose: every method here is a Qdrant concept,
// and an abstraction would only be a worse copy of this API.
package qdrant

import (
	"context"
	"fmt"
	"strings"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/Below-Z3R0/argos/internal/types"
)

// Store is a thin, typed wrapper around the client.
type Store struct {
	client        *qdrant.Client
	host          string
	restURL       string
	snapshotsPath string
}

// Point is the shared point type.
type Point = types.Point

// Hit is the shared search result type.
type Hit = types.Hit

// New connects to a Qdrant instance over gRPC with the conventional REST
// endpoint (host:6333) and snapshot path (/qdrant/snapshots).
func New(host string, port int) (*Store, error) {
	return NewWithREST(host, port, "", "")
}

// NewWithREST connects and records the REST endpoint and the snapshot
// directory as Qdrant sees it. Empty values keep the conventional defaults;
// a remote or re-mapped Qdrant needs both to restore snapshots.
func NewWithREST(host string, port int, restURL, snapshotsPath string) (*Store, error) {
	if host == "" {
		host = "127.0.0.1"
	}
	if port == 0 {
		port = 6334
	}
	c, err := qdrant.NewClient(&qdrant.Config{Host: host, Port: port, SkipCompatibilityCheck: true})
	if err != nil {
		return nil, fmt.Errorf("qdrant: connect: %w", err)
	}
	if snapshotsPath == "" {
		snapshotsPath = "/qdrant/snapshots"
	}
	return &Store{
		client:        c,
		host:          host,
		restURL:       strings.TrimSuffix(restURL, "/"),
		snapshotsPath: strings.TrimSuffix(snapshotsPath, "/"),
	}, nil
}

// Close releases the connection.
func (s *Store) Close() error { return s.client.Close() }

// EnsureCollection creates the collection when it does not exist and verifies
// that an existing one uses the same dimension. Mixing dimensions in one
// collection is not something Qdrant can recover from, so it fails loudly.
func (s *Store) EnsureCollection(ctx context.Context, name string, dim int, quantization string) (bool, error) {
	info, err := s.client.GetCollectionsClient().Get(ctx, &qdrant.GetCollectionInfoRequest{CollectionName: name})
	if err == nil {
		params := info.GetResult().GetConfig().GetParams()
		vectors := params.GetVectorsConfig().GetParamsMap().GetMap()
		dense, ok := vectors["dense"]
		if !ok {
			return false, fmt.Errorf("qdrant: collection %s uses the old unnamed schema; drop it or use another collection (index.collection)", name)
		}
		if size := int(dense.GetSize()); size != 0 && size != dim {
			return false, fmt.Errorf("qdrant: collection %s has %d dims, this index uses %d; use another collection (index.collection)", name, size, dim)
		}
		if params.GetSparseVectorsConfig().GetMap()["sparse"] == nil {
			return false, fmt.Errorf("qdrant: collection %s has no sparse vector; use another collection (index.collection)", name)
		}
		return false, nil
	}
	if status.Code(err) != codes.NotFound {
		return false, fmt.Errorf("qdrant: collection %s: %w", name, err)
	}

	create := &qdrant.CreateCollection{
		CollectionName: name,
		VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
			"dense": {Size: uint64(dim), Distance: qdrant.Distance_Cosine},
		}),
		SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
			"sparse": {Modifier: ptr(qdrant.Modifier_Idf)},
		}),
	}
	if quantization == "int8" {
		create.QuantizationConfig = &qdrant.QuantizationConfig{
			Quantization: &qdrant.QuantizationConfig_Scalar{Scalar: &qdrant.ScalarQuantization{
				Type:      qdrant.QuantizationType_Int8,
				Quantile:  ptr(float32(0.99)),
				AlwaysRam: ptr(true),
			}},
		}
	}
	if err := s.client.CreateCollection(ctx, create); err != nil {
		return false, fmt.Errorf("qdrant: create %s: %w", name, err)
	}
	return true, nil
}

// Health reports whether the server answers, regardless of collections.
func (s *Store) Health(ctx context.Context) error {
	_, err := s.client.CollectionExists(ctx, "__argos_health")
	return err
}

// CollectionDim returns the vector size of an existing collection.
func (s *Store) CollectionDim(ctx context.Context, name string) (int, error) {
	info, err := s.client.GetCollectionsClient().Get(ctx, &qdrant.GetCollectionInfoRequest{CollectionName: name})
	if err != nil {
		return 0, err
	}
	return int(info.GetResult().GetConfig().GetParams().GetVectorsConfig().GetParamsMap().GetMap()["dense"].GetSize()), nil
}

// EnsurePayloadIndexes creates the indexes the filter paths depend on.
// Qdrant recommends creating them before uploading, not after.
func (s *Store) EnsurePayloadIndexes(ctx context.Context, collection string) error {
	fields := []struct {
		name  string
		ftype qdrant.FieldType
	}{
		{"path", qdrant.FieldType_FieldTypeKeyword},
		{"file_hash", qdrant.FieldType_FieldTypeKeyword},
		{"mtime", qdrant.FieldType_FieldTypeInteger},
		{"tags", qdrant.FieldType_FieldTypeKeyword},
		{"lang", qdrant.FieldType_FieldTypeKeyword},
	}
	for _, f := range fields {
		_, err := s.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName: collection,
			FieldName:      f.name,
			FieldType:      f.ftype.Enum(),
			Wait:           ptr(true),
		})
		if err != nil && !strings.Contains(err.Error(), "already exists") {
			return fmt.Errorf("qdrant: index %s on %s: %w", f.name, collection, err)
		}
	}
	return nil
}

// Upsert writes points and waits for the change to be applied.
func (s *Store) Upsert(ctx context.Context, collection string, points []Point) error {
	return s.UpsertWithWait(ctx, collection, points, true)
}

// UpsertWithWait controls whether the write blocks until indexing finishes.
// Bulk ingestion uses false and waits once at the end: wait=true is 3–5x slower
// per batch.
func (s *Store) UpsertWithWait(ctx context.Context, collection string, points []Point, wait bool) error {
	if len(points) == 0 {
		return nil
	}
	batch := make([]*qdrant.PointStruct, len(points))
	for i, p := range points {
		vectors := map[string]*qdrant.Vector{
			"dense": {Vector: &qdrant.Vector_Dense{Dense: &qdrant.DenseVector{Data: p.Vector}}},
		}
		if len(p.SparseIndices) > 0 {
			vectors["sparse"] = &qdrant.Vector{Vector: &qdrant.Vector_Sparse{Sparse: &qdrant.SparseVector{
				Indices: p.SparseIndices, Values: p.SparseValues,
			}}}
		}
		batch[i] = &qdrant.PointStruct{
			Id:      qdrant.NewID(p.ID),
			Vectors: qdrant.NewVectorsMap(vectors),
			Payload: qdrant.NewValueMap(p.Payload),
		}
	}
	_, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: collection,
		Wait:           ptr(wait),
		Points:         batch,
	})
	if err != nil {
		return fmt.Errorf("qdrant: upsert into %s: %w", collection, err)
	}
	return nil
}

// Search returns the closest points to a vector.
func (s *Store) Search(ctx context.Context, collection string, vector []float32, limit int, filter map[string]string) ([]Hit, error) {
	req := &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQueryDense(vector),
		Using:          ptr("dense"),
		Limit:          ptr(uint64(limit)),
		WithPayload:    qdrant.NewWithPayload(true),
	}
	if len(filter) > 0 {
		req.Filter = filterFor(filter)
	}
	res, err := s.client.Query(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("qdrant: search %s: %w", collection, err)
	}
	out := make([]Hit, 0, len(res))
	for _, p := range res {
		out = append(out, Hit{ID: p.GetId().GetUuid(), Score: p.GetScore(), Payload: payloadOf(p.GetPayload())})
	}
	return out, nil
}

// SearchHybrid fuses dense and sparse candidates with RRF. With an empty
// sparse vector it falls back to dense-only.
func (s *Store) SearchHybrid(ctx context.Context, collection string, dense []float32, sparseIdx []uint32, sparseVal []float32, limit int, filter map[string]string) ([]Hit, error) {
	if len(sparseIdx) == 0 {
		return s.Search(ctx, collection, dense, limit, filter)
	}
	var flt *qdrant.Filter
	if len(filter) > 0 {
		flt = filterFor(filter)
	}
	req := &qdrant.QueryPoints{
		CollectionName: collection,
		Prefetch: []*qdrant.PrefetchQuery{
			{Query: qdrant.NewQueryDense(dense), Using: ptr("dense"), Limit: ptr(uint64(limit)), Filter: flt},
			{Query: qdrant.NewQuerySparse(sparseIdx, sparseVal), Using: ptr("sparse"), Limit: ptr(uint64(limit)), Filter: flt},
		},
		Query:       qdrant.NewQueryFusion(qdrant.Fusion_RRF),
		Limit:       ptr(uint64(limit)),
		WithPayload: qdrant.NewWithPayload(true),
		Filter:      flt,
	}
	res, err := s.client.Query(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("qdrant: hybrid search %s: %w", collection, err)
	}
	out := make([]Hit, 0, len(res))
	for _, p := range res {
		out = append(out, Hit{ID: p.GetId().GetUuid(), Score: p.GetScore(), Payload: payloadOf(p.GetPayload())})
	}
	return out, nil
}

// DeleteByPayload removes every point whose field matches the value.
func (s *Store) DeleteByPayload(ctx context.Context, collection, field, value string) error {
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: collection,
		Wait:           ptr(true),
		Points:         qdrant.NewPointsSelectorFilter(filterFor(map[string]string{field: value})),
	})
	if err != nil {
		return fmt.Errorf("qdrant: delete from %s: %w", collection, err)
	}
	return nil
}

// DeleteByIDs removes specific points, used when chunk content changes.
func (s *Store) DeleteByIDs(ctx context.Context, collection string, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	// gRPC rejects oversized messages; 256 ids per call is well under the limit.
	for start := 0; start < len(ids); start += 256 {
		end := start + 256
		if end > len(ids) {
			end = len(ids)
		}
		if err := s.deleteIDs(ctx, collection, ids[start:end]); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) deleteIDs(ctx context.Context, collection string, ids []string) error {
	pointIDs := make([]*qdrant.PointId, len(ids))
	for i, id := range ids {
		pointIDs[i] = qdrant.NewID(id)
	}
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: collection,
		Wait:           ptr(true),
		Points:         qdrant.NewPointsSelectorIDs(pointIDs),
	})
	if err != nil {
		return fmt.Errorf("qdrant: delete ids from %s: %w", collection, err)
	}
	return nil
}

// Count returns the number of points matching the filter.
func (s *Store) Count(ctx context.Context, collection string, filter map[string]string) (uint64, error) {
	req := &qdrant.CountPoints{CollectionName: collection, Exact: ptr(true)}
	if len(filter) > 0 {
		req.Filter = filterFor(filter)
	}
	n, err := s.client.Count(ctx, req)
	if err != nil {
		return 0, fmt.Errorf("qdrant: count %s: %w", collection, err)
	}
	return n, nil
}

// DeleteCollection drops a collection and everything in it. A missing
// collection is not an error: the base may have been removed out of band.
func (s *Store) DeleteCollection(ctx context.Context, name string) error {
	err := s.client.DeleteCollection(ctx, name)
	if status.Code(err) == codes.NotFound {
		return nil
	}
	return err
}

func filterFor(fields map[string]string) *qdrant.Filter {
	must := make([]*qdrant.Condition, 0, len(fields))
	for k, v := range fields {
		must = append(must, qdrant.NewMatch(k, v))
	}
	return &qdrant.Filter{Must: must}
}

// payloadOf keeps the payload types: a token count is an integer, not a
// string, and losing that forces the caller to re-parse everything.
func payloadOf(in map[string]*qdrant.Value) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		if v == nil {
			continue
		}
		switch kind := v.Kind.(type) {
		case *qdrant.Value_StringValue:
			out[k] = kind.StringValue
		case *qdrant.Value_IntegerValue:
			out[k] = kind.IntegerValue
		case *qdrant.Value_DoubleValue:
			out[k] = kind.DoubleValue
		case *qdrant.Value_BoolValue:
			out[k] = kind.BoolValue
		case *qdrant.Value_ListValue:
			// Tags come back as a list; dropping them silently broke filters.
			items := make([]any, 0, len(kind.ListValue.GetValues()))
			for _, item := range kind.ListValue.GetValues() {
				if item != nil && item.GetStringValue() != "" {
					items = append(items, item.GetStringValue())
				}
			}
			out[k] = items
		}
	}
	return out
}

func ptr[T any](v T) *T { return &v }
