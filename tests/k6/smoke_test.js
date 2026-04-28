/**
 * k6 Smoke Test — TC-K6-01
 * Verifies the RL agent service is healthy and responds within 200ms
 * under a single virtual user.
 *
 * Run:
 *   k6 run k6/smoke_test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const healthLatency = new Trend('health_latency_ms');
const healthSuccess = new Rate('health_success_rate');

export const options = {
  vus: 1,
  duration: '30s',
  thresholds: {
    // NFR1: response time under 200ms for health checks
    'health_latency_ms': ['p(95)<200'],
    // NFR3: 100% success rate for basic availability
    'health_success_rate': ['rate>0.99'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://rl-agent.dev-sachin.co.uk';

export default function () {
  const res = http.get(`${BASE_URL}/health`);

  healthLatency.add(res.timings.duration);
  healthSuccess.add(res.status === 200);

  check(res, {
    'status is 200':       (r) => r.status === 200,
    'status is healthy':   (r) => JSON.parse(r.body).status === 'healthy',
    'response under 200ms':(r) => r.timings.duration < 200,
  });

  sleep(1);
}
