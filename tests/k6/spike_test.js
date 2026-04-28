/**
 * k6 Spike Test — TC-K6-04
 * Simulates a sudden traffic spike to verify NFR3 (reliability) and
 * NFR2 (scalability). Jumps from 1 VU to 50 VU instantaneously.
 *
 * Run:
 *   k6 run k6/spike_test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const spikeSuccess = new Rate('spike_success_rate');
const spikeLatency = new Trend('spike_latency_ms');

export const options = {
  stages: [
    { duration: '10s', target: 1  },  // baseline
    { duration: '5s',  target: 50 },  // sudden spike
    { duration: '30s', target: 50 },  // hold spike
    { duration: '10s', target: 1  },  // recovery
  ],
  thresholds: {
    // NFR3: error rate must stay below 5% even during spike
    'spike_success_rate': ['rate>0.95'],
    // NFR1: p99 must stay under 2s even under spike load
    'spike_latency_ms':   ['p(99)<2000'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://rl-agent.dev-sachin.co.uk';

export default function () {
  const res = http.post(
    `${BASE_URL}/predict`,
    JSON.stringify({
      deployment_name: 'spike-test-app',
      namespace: 'default',
      training_mode: false,
      metrics: {
        cpu_usage:    0.7,
        memory_usage: 3.0,
        request_rate: 200,
        latency_p50:  0.4,
        latency_p95:  0.7,
        latency_p99:  0.9,
        replicas:     2,
        error_rate:   0.0,
        pod_pending:  0,
        pod_ready:    2,
        cpu_trend_1m: 0.05,
        cpu_trend_5m: 0.03,
        request_trend:10.0,
        hour:         14,
        day_of_week:  2,
        is_weekend:   false,
        is_peak_hour: true,
      },
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );

  spikeSuccess.add(res.status === 200);
  spikeLatency.add(res.timings.duration);

  check(res, {
    'no 5xx errors':        (r) => r.status < 500,
    'response is valid':    (r) => {
      try {
        const b = JSON.parse(r.body);
        return b.success === true;
      } catch { return false; }
    },
  });

  sleep(0.1);
}
