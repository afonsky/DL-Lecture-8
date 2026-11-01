<script setup lang="ts">
import { computed, ref } from 'vue'

type Point = { x: number, y: number }
type Optimizer = 'SGD' | 'Momentum' | 'RMSProp' | 'Adam'

const optimizer = ref<Optimizer>('Momentum')
const lr = ref(0.4)
const beta = ref(0.5)
const path = ref<Point[]>([{ x: -5, y: -2 }])
const velocity = ref<Point>({ x: 0, y: 0 })
const second = ref<Point>({ x: 0, y: 0 })
const mean = ref<Point>({ x: 0, y: 0 })

const levels = [0.5, 2, 5, 10, 20, 30]
const current = computed(() => path.value[path.value.length - 1])
const steps = computed(() => path.value.length - 1)
const loss = computed(() => 0.1 * current.value.x ** 2 + 2 * current.value.y ** 2)
const diverged = computed(() => !Number.isFinite(loss.value) || Math.abs(current.value.x) > 20 || Math.abs(current.value.y) > 20)
const sx = (x: number) => 20 + (x + 5.5) * 65
const sy = (y: number) => 270 - (y + 3) * 60
const pathText = computed(() => path.value.map((p, i) => `${i ? 'L' : 'M'} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(' '))
const formula = computed(() => ({
  SGD: 'θ ← θ − ηg',
  Momentum: 'v ← βv + g;  θ ← θ − ηv',
  RMSProp: 's ← 0.9s + 0.1g²;  θ ← θ − ηg / √(s + ε)',
  Adam: 'm, s ← moving averages;  θ ← θ − ηm̂ / (√ŝ + ε)',
}[optimizer.value]))

function reset() {
  path.value = [{ x: -5, y: -2 }]
  velocity.value = { x: 0, y: 0 }
  second.value = { x: 0, y: 0 }
  mean.value = { x: 0, y: 0 }
}

function selectOptimizer(name: Optimizer) {
  optimizer.value = name
  reset()
}

function step() {
  if (steps.value >= 50 || diverged.value) return
  const p = current.value
  const g = { x: 0.2 * p.x, y: 4 * p.y }
  let update = { ...g }
  const t = steps.value + 1
  if (optimizer.value === 'Momentum') {
    velocity.value = { x: beta.value * velocity.value.x + g.x, y: beta.value * velocity.value.y + g.y }
    update = velocity.value
  } else if (optimizer.value === 'RMSProp') {
    second.value = { x: 0.9 * second.value.x + 0.1 * g.x ** 2, y: 0.9 * second.value.y + 0.1 * g.y ** 2 }
    update = { x: g.x / Math.sqrt(second.value.x + 1e-6), y: g.y / Math.sqrt(second.value.y + 1e-6) }
  } else if (optimizer.value === 'Adam') {
    mean.value = { x: 0.9 * mean.value.x + 0.1 * g.x, y: 0.9 * mean.value.y + 0.1 * g.y }
    second.value = { x: 0.999 * second.value.x + 0.001 * g.x ** 2, y: 0.999 * second.value.y + 0.001 * g.y ** 2 }
    const mh = { x: mean.value.x / (1 - 0.9 ** t), y: mean.value.y / (1 - 0.9 ** t) }
    const sh = { x: second.value.x / (1 - 0.999 ** t), y: second.value.y / (1 - 0.999 ** t) }
    update = { x: mh.x / (Math.sqrt(sh.x) + 1e-8), y: mh.y / (Math.sqrt(sh.y) + 1e-8) }
  }
  path.value = [...path.value, { x: p.x - lr.value * update.x, y: p.y - lr.value * update.y }]
}

function run(count: number) {
  for (let i = 0; i < count; i++) step()
}

function fmt(v: number) {
  if (!Number.isFinite(v)) return '∞'
  if (v >= 1000) return v.toExponential(1)
  return v.toFixed(3)
}
</script>

<template>
  <div class="op">
    <div class="op-controls">
      <button v-for="name in ['SGD', 'Momentum', 'RMSProp', 'Adam'] as Optimizer[]" :key="name"
              :class="{ 'op-sel': optimizer === name }" @click="selectOptimizer(name)">{{ name }}</button>
      <span class="op-label">η</span>
      <button v-for="v in [0.05, 0.1, 0.2, 0.4, 0.6]" :key="v" :class="{ 'op-sel': lr === v }" @click="lr = v; reset()">{{ v }}</button>
      <template v-if="optimizer === 'Momentum'">
        <span class="op-label">β</span>
        <button v-for="v in [0, 0.5, 0.9]" :key="v" :class="{ 'op-sel': beta === v }" @click="beta = v; reset()">{{ v }}</button>
      </template>
      <button class="op-action" @click="step">+1 step</button>
      <button class="op-action" @click="run(20)">+20 steps</button>
      <button class="op-action" @click="reset">reset</button>
    </div>
    <div class="op-body">
      <svg viewBox="0 0 465 285" class="op-plot" aria-label="Optimizer trajectory on an elongated quadratic loss">
        <ellipse v-for="level in levels" :key="level" :cx="sx(0)" :cy="sy(0)"
                 :rx="Math.sqrt(level / 0.1) * 65" :ry="Math.sqrt(level / 2) * 60"
                 fill="none" stroke="#b8c9d6" stroke-width="1" />
        <path :d="pathText" fill="none" stroke="#e76f51" stroke-width="2.5" />
        <circle v-for="(p, i) in path" :key="i" :cx="sx(p.x)" :cy="sy(p.y)" r="3.2" fill="#e76f51" />
        <path :d="`M ${sx(0) - 7} ${sy(0)} L ${sx(0) + 7} ${sy(0)} M ${sx(0)} ${sy(0) - 7} L ${sx(0)} ${sy(0) + 7}`" stroke="#222" stroke-width="2" />
        <text x="12" y="18" class="op-axis">x₂: steep direction ↑</text>
        <text x="310" y="277" class="op-axis">x₁: flat direction →</text>
      </svg>
      <div class="op-readout">
        <div class="op-formula">{{ formula }}</div>
        <div>step <b>{{ steps }}</b> / 50</div>
        <div>x = ({{ fmt(current.x) }}, {{ fmt(current.y) }})</div>
        <div>loss = <b>{{ fmt(loss) }}</b></div>
        <div v-if="diverged" class="op-warning">The trajectory left the plotted region: divergence.</div>
        <div v-else class="op-muted">f(x₁, x₂) = 0.1x₁² + 2x₂²<br>Same start, same gradient. Only the update rule changes.</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.op { color: #161616; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 13px; }
.op-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 5px; margin-bottom: 7px; }
.op-controls button { border: 1px solid #b9b9b9; background: white; border-radius: 4px; padding: 2px 7px; color: #444; }
.op-controls button.op-sel { color: white; background: #2a78d6; border-color: #2a78d6; }
.op-controls .op-action { background: #f2f2f2; margin-left: 4px; }
.op-label { color: #777; margin-left: 8px; }
.op-body { display: grid; grid-template-columns: 3fr 1.3fr; gap: 14px; align-items: center; }
.op-plot { width: 100%; height: 315px; border: 1px solid #ddd; background: #fcfcfc; }
.op-axis { font-size: 11px; fill: #666; }
.op-readout { display: flex; flex-direction: column; gap: 8px; font-variant-numeric: tabular-nums; }
.op-formula { font-size: 12px; color: #064789; min-height: 36px; }
.op-muted { color: #777; font-size: 12px; line-height: 1.35; }
.op-warning { color: #b42318; font-weight: 600; }
</style>
