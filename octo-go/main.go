// octo-go/main.go
//
// OctoTetrahedral Go Parallel Client
// ===================================
//
// Demonstrates compounding cognitive cohesion from Go by calling the Rust
// orchestrator (which in turn fans out to all Python limbs in parallel and
// runs Block AttnRes natively).
//
// Architecture recap:
//
//   Go client                Rust orchestrator            Python limb server
//   ─────────────────────    ────────────────────────     ────────────────────
//   goroutine fan-out  →  POST /infer   →   POST /encode
//                                           POST /limb/*  (×13, parallel)
//                                           Block AttnRes (native Rust)
//                      ←  JSON response  ←  compound result
//   compound & display
//
// Usage:
//   go run main.go --rust http://localhost:8766 --tokens 42,17,93,7
//   go run main.go --bench 50   # send 50 parallel infer requests

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Wire types
// ─────────────────────────────────────────────────────────────────────────────

type InferRequest struct {
	InputIDs []int `json:"input_ids"`
}

type LimbResult struct {
	Limb      string  `json:"limb"`
	Confidence float64 `json:"confidence"`
	Shape      []int   `json:"shape"`
	ElapsedMs  float64 `json:"elapsed_ms"`
}

type InferResponse struct {
	Limbs            []LimbResult `json:"limbs"`
	RustBraidApplied bool         `json:"rust_braid_applied"`
	TotalMs          float64      `json:"total_ms"`
	NBraidBlocks     int          `json:"n_braid_blocks"`
}

// ─────────────────────────────────────────────────────────────────────────────
// Client
// ─────────────────────────────────────────────────────────────────────────────

type OctoClient struct {
	rustURL string
	http    *http.Client
}

func newClient(rustURL string, timeout time.Duration) *OctoClient {
	return &OctoClient{
		rustURL: rustURL,
		http:    &http.Client{Timeout: timeout},
	}
}

func (c *OctoClient) Infer(ctx context.Context, tokenIDs []int) (*InferResponse, error) {
	body, err := json.Marshal(InferRequest{InputIDs: tokenIDs})
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.rustURL+"/infer", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("POST /infer: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(raw))
	}

	var result InferResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}
	return &result, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Display helpers
// ─────────────────────────────────────────────────────────────────────────────

func displayResult(resp *InferResponse, wallMs float64) {
	fmt.Printf("\n╔═══════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║        OctoTetrahedral Parallel Inference Result          ║\n")
	fmt.Printf("╠═══════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Limbs processed : %-38d║\n", len(resp.Limbs))
	fmt.Printf("║  Rust braid      : %-38v║\n", resp.RustBraidApplied)
	fmt.Printf("║  Braid blocks    : %-38d║\n", resp.NBraidBlocks)
	fmt.Printf("║  Server total    : %-34.1f ms ║\n", resp.TotalMs)
	fmt.Printf("║  Go wall-clock   : %-34.1f ms ║\n", wallMs)
	fmt.Printf("╠═══════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  %-18s  %8s  %12s  %8s  ║\n", "Limb", "Conf", "Shape", "ms")
	fmt.Printf("╠═══════════════════════════════════════════════════════════╣\n")

	// Sort by confidence desc for display
	sorted := make([]LimbResult, len(resp.Limbs))
	copy(sorted, resp.Limbs)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Confidence > sorted[j].Confidence
	})

	for _, l := range sorted {
		shapeStr := fmt.Sprintf("%v", l.Shape)
		fmt.Printf("║  %-18s  %8.4f  %12s  %8.1f  ║\n",
			l.Limb, l.Confidence, shapeStr, l.ElapsedMs)
	}
	fmt.Printf("╚═══════════════════════════════════════════════════════════╝\n\n")
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: N parallel infer requests via goroutines
// ─────────────────────────────────────────────────────────────────────────────

type benchStats struct {
	n       int
	ok      int64
	failed  int64
	totalMs []float64
	mu      sync.Mutex
}

func runBench(client *OctoClient, n int, tokenIDs []int) {
	fmt.Printf("Running benchmark: %d parallel infer requests…\n\n", n)

	stats := &benchStats{n: n, totalMs: make([]float64, 0, n)}
	var wg sync.WaitGroup

	t0 := time.Now()
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			t := time.Now()
			resp, err := client.Infer(ctx, tokenIDs)
			elapsed := time.Since(t).Seconds() * 1000

			if err != nil {
				atomic.AddInt64(&stats.failed, 1)
				fmt.Fprintf(os.Stderr, "[%d] ERROR: %v\n", idx, err)
				return
			}
			atomic.AddInt64(&stats.ok, 1)
			_ = resp

			stats.mu.Lock()
			stats.totalMs = append(stats.totalMs, elapsed)
			stats.mu.Unlock()
		}(i)
	}
	wg.Wait()
	wallSec := time.Since(t0).Seconds()

	// Compute percentiles
	sort.Float64s(stats.totalMs)
	p50 := percentile(stats.totalMs, 50)
	p95 := percentile(stats.totalMs, 95)
	p99 := percentile(stats.totalMs, 99)

	fmt.Printf("Benchmark complete\n")
	fmt.Printf("  Requests  : %d total, %d ok, %d failed\n", n, stats.ok, stats.failed)
	fmt.Printf("  Wall time : %.2f s\n", wallSec)
	fmt.Printf("  Throughput: %.1f req/s\n", float64(stats.ok)/wallSec)
	fmt.Printf("  Latency   : p50=%.1fms  p95=%.1fms  p99=%.1fms\n", p50, p95, p99)
}

func percentile(sorted []float64, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * float64(p) / 100.0)
	return sorted[idx]
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	rustURL   := flag.String("rust",   "http://localhost:8766", "Rust orchestrator URL")
	benchN    := flag.Int("bench",     0,    "Run N parallel infer requests (benchmark mode)")
	timeout   := flag.Duration("timeout", 120*time.Second, "Per-request timeout")
	flag.Parse()

	// Default token IDs — would come from a real tokeniser in production
	tokenIDs := []int{42, 17, 93, 7, 105, 3, 99, 11, 28, 64, 2, 50, 76, 33, 8, 19}

	client := newClient(*rustURL, *timeout)

	// Healthcheck
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	hReq, _ := http.NewRequestWithContext(ctx, "GET", *rustURL+"/healthz", nil)
	hResp, err := client.http.Do(hReq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Rust orchestrator not reachable at %s: %v\n", *rustURL, err)
		fmt.Fprintf(os.Stderr, "Start it with:  cd octo-parallel-rs && cargo run\n")
		os.Exit(1)
	}
	hResp.Body.Close()
	fmt.Printf("✓ Rust orchestrator healthy at %s\n", *rustURL)

	if *benchN > 0 {
		runBench(client, *benchN, tokenIDs)
		return
	}

	// Single inference with display
	fmt.Printf("Sending infer request (tokens=%v)…\n", tokenIDs)
	t0 := time.Now()
	resp, err := client.Infer(context.Background(), tokenIDs)
	wallMs := time.Since(t0).Seconds() * 1000
	if err != nil {
		fmt.Fprintf(os.Stderr, "Infer failed: %v\n", err)
		os.Exit(1)
	}
	displayResult(resp, wallMs)
}
