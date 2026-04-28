/**
 * k6 Load Test — TC-K6-02 and TC-K6-03
 * Tests /predict under sustained concurrent load (NFR1, NFR2).
 *
 * Stages:
 *   0→10 VU over 30s  — ramp up
 *   10 VU for 60s     — sustained load
 *   10→0 VU over 15s  — ramp down
 *
 * Run:
 *   k6 run k6/predict_load_test.js
 *   k6 run -e BASE_URL=https://rl-agent.dev-sachin.co.uk k6/predict_load_test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

const predictLatency   = new Trend('predict_latency_ms');
const predictSuccess   = new Rate('predict_success_rate');
const scalingDecisions = new Counter('scaling_decisions_total');

export const options = {
  stages: [
    { duration: '30s', target: 10 },
    { duration: '60s', target: 10 },
    { duration: '15s', target: 0  },
  ],
  thresholds: {
    // NFR1: 95% of predict calls must complete within 500ms
    'predict_latency_ms':  ['p(95)<500'],
    // NFR3: at least 99% success rate under load
    'predict_success_rate':['rate>0.99'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://rl-agent.dev-sachin.co.uk';

const DEPLOYMENTS = ['app-a', 'app-b', 'app-c'];

function randomDeployment() {
  return DEPLOYMENTS[Math.floor(Math.random() * DEPLOYMENTS.length)];
}

function makePayload(deployment, latency, replicas) {
  return JSON.stringify({
    deployment_name: deployment,
    namespace: 'default',
    training_mode: false,
    metrics: {
      cpu_usage:    0.3 + Math.random() * 0.4,
      memory_usage: 1.5 + Math.random() * 1.0,
      request_rate: 50  + Math.random() * 100,
      latency_p50:  latency * 0.7,
      latency_p95:  latency,
      latency_p99:  latency * 1.3,
      replicas:     replicas,
      error_rate:   Math.random() * 0.005,
      pod_pending:  0,
      pod_ready:    replicas,
      cpu_trend_1m: 0.0,
      cpu_trend_5m: 0.0,
      request_trend:0.0,
      hour:         new Date().getHours(),
      day_of_week:  new Date().getDay(),
      is_weekend:   false,
      is_peak_hour: true,
    },
  });
}

export default function () {
  const deployment = randomDeployment();
  const latency    = 0.1 + Math.random() * 0.8;
  const replicas   = Math.floor(1 + Math.random() * 4);

  const res = http.post(
    `${BASE_URL}/predict`,
    makePayload(deployment, latency, replicas),
    { headers: { 'Content-Type': 'application/json' } }
  );

  predictLatency.add(res.timings.duration);
  predictSuccess.add(res.status === 200);

  const ok = check(res, {
    'status 200':            (r) => r.status === 200,
    'success true':          (r) => {
      try { return JSON.parse(r.body).success === true; } catch (_) { return false; }
    },
    'action in valid range': (r) => {
      try {
        const b = JSON.parse(r.body);
        return b.action === 0 || b.action === 1 || b.action === 2;
      } catch (_) { return false; }
    },
    'confidence in [0,1]':   (r) => {
      try {
        const b = JSON.parse(r.body);
        return b.confidence >= 0.0 && b.confidence <= 1.0;
      } catch (_) { return false; }
    },
    'response under 500ms':  (r) => r.timings.duration < 500,
  });

  if (ok) scalingDecisions.add(1);
  sleep(0.5 + Math.random() * 0.5);
}
