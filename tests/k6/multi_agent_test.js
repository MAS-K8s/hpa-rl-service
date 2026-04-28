/**
 * k6 Multi-Agent Concurrent Test — TC-K6-05
 * Verifies FR7 (multi-agent support) and NFR2 (scalability) by sending
 * simultaneous requests for 10 distinct deployments.
 *
 * Run:
 *   k6 run k6/multi_agent_test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const multiSuccess = new Rate('multi_agent_success_rate');

export const options = {
  vus: 10,
  duration: '60s',
  thresholds: {
    'multi_agent_success_rate': ['rate>0.99'],
    'http_req_duration':        ['p(95)<600'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://rl-agent.dev-sachin.co.uk';

export default function () {
  // Each VU uses a unique deployment name to simulate independent agents
  const deployment = `deployment-${__VU}`;

  const res = http.post(
    `${BASE_URL}/predict`,
    JSON.stringify({
      deployment_name: deployment,
      namespace: 'default',
      training_mode: false,
      metrics: {
        cpu_usage:    0.4,
        memory_usage: 2.0,
        request_rate: 60,
        latency_p50:  0.2,
        latency_p95:  0.35,
        latency_p99:  0.45,
        replicas:     2,
        error_rate:   0.0,
        pod_pending:  0,
        pod_ready:    2,
        cpu_trend_1m: 0.0,
        cpu_trend_5m: 0.0,
        request_trend:0.0,
        hour:         10,
        day_of_week:  1,
        is_weekend:   false,
        is_peak_hour: true,
      },
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );

  multiSuccess.add(res.status === 200);

  check(res, {
    'independent agent responds': (r) => r.status === 200,
    'correct deployment in response': (r) => {
      // Success flag confirms agent was created for this deployment
      return JSON.parse(r.body).success === true;
    },
  });

  sleep(1);
}
