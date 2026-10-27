import type { StarSample } from '@site/src/components/render/stars'
import * as d3 from 'd3'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

/** The directions the camera looks from: face-on, edge-on, and tilted by about 30 degrees from edge-on. */
export type View = 'face' | 'edge' | 'tilt'

const DIRECTIONS: Record<View, readonly [number, number, number]> = {
  face: [0, -1e-3, 1],
  edge: [0, -1, 0],
  tilt: [0, -0.87, 0.5]
}

/** The vertical field of view of the camera, which the length scale needs. */
export const FIELD_OF_VIEW_DEG = 40
const TRANSITION_MS = 1400
const ROTATE_SPEED = 0.5
const POINT_SIZE = 1.15
const INTENSITY = 0.3

const VERTEX_SHADER = (stops: number): string => `
attribute float age;
uniform float uAgeMin;
uniform float uAgeMax;
uniform float uSize;
uniform float uPixelRatio;
uniform vec3 uColors[${stops}];
uniform float uStops[${stops}];
varying vec3 vColor;

vec3 palette(float a) {
  vec3 color = uColors[0];
  for (int i = 0; i < ${stops - 1}; i++) {
    float t = clamp((a - uStops[i]) / (uStops[i + 1] - uStops[i]), 0.0, 1.0);
    color = mix(color, uColors[i + 1], t);
  }
  return color;
}

void main() {
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * viewPosition;
  float inside = step(uAgeMin, age) * step(age, uAgeMax);
  vColor = palette(age);
  gl_PointSize = inside * uSize * uPixelRatio * 100.0 / -viewPosition.z;
}
`

const FRAGMENT_SHADER = `
uniform float uIntensity;
varying vec3 vColor;

void main() {
  vec2 offset = gl_PointCoord - vec2(0.5);
  float r2 = 4.0 * dot(offset, offset);
  if (r2 > 1.0) discard;
  float glow = exp(-3.5 * r2) - exp(-3.5);
  gl_FragColor = vec4(vColor * glow * uIntensity, 1.0);
  #include <colorspace_fragment>
}
`

/**
 * A three.js scene of star particles drawn as soft points that add their light,
 * colored by age, with the camera on an orbit around the spin axis of the disk.
 */
export default class StarScene {
  private readonly renderer: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera: THREE.PerspectiveCamera
  private readonly controls: OrbitControls
  private points: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial> | null = null
  private ageRange: readonly [number, number] = [0, 14]
  private distance: number
  private frame: number | null = null
  private transition: { from: THREE.Vector3; to: THREE.Vector3; start: number } | null = null

  constructor(canvas: HTMLCanvasElement, distance: number) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.setClearColor(0x000000, 1)
    this.distance = distance
    this.camera = new THREE.PerspectiveCamera(FIELD_OF_VIEW_DEG, 1, 0.1, 1000)
    this.camera.up.set(0, 0, 1)
    this.camera.position.set(...this.target('tilt').toArray())
    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.enableZoom = false
    this.controls.enablePan = false
    this.controls.autoRotateSpeed = ROTATE_SPEED
  }

  /** The camera position that looks from `view` at the current distance. */
  private target(view: View): THREE.Vector3 {
    return new THREE.Vector3(...DIRECTIONS[view]).normalize().multiplyScalar(this.distance)
  }

  setSize(width: number, height: number): void {
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }

  /** Replace the stars by `sample`, with its age palette. */
  setSample(sample: StarSample): void {
    this.disposePoints()
    const { ageStopsGyr, ageColors } = sample.meta
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(sample.positions, 3))
    geometry.setAttribute('age', new THREE.BufferAttribute(sample.ages, 1))
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER(ageStopsGyr.length),
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        uAgeMin: { value: this.ageRange[0] },
        uAgeMax: { value: this.ageRange[1] },
        uSize: { value: POINT_SIZE },
        uPixelRatio: { value: this.renderer.getPixelRatio() },
        uIntensity: { value: INTENSITY },
        uStops: { value: [...ageStopsGyr] },
        uColors: {
          value: ageColors.map((color) => new THREE.Color().setStyle(color, THREE.SRGBColorSpace))
        }
      },
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
      transparent: true
    })
    this.points = new THREE.Points(geometry, material)
    this.scene.add(this.points)
  }

  /** Show only the stars with ages (Gyr) within `range`. */
  setAgeRange(range: readonly [number, number]): void {
    this.ageRange = range
    if (this.points !== null) {
      this.points.material.uniforms.uAgeMin.value = range[0]
      this.points.material.uniforms.uAgeMax.value = range[1]
    }
  }

  /** Move the camera to `distance` (kpc) from the center, along its current direction. */
  setDistance(distance: number): void {
    this.distance = distance
    this.camera.position.setLength(distance)
  }

  /** Fly the camera to look from `view`, or jump there when `animate` is false. */
  setView(view: View, animate: boolean): void {
    const to = this.target(view)
    if (animate) {
      this.transition = { from: this.camera.position.clone(), to, start: performance.now() }
    } else {
      this.camera.position.copy(to)
    }
  }

  setRotate(rotate: boolean): void {
    this.controls.autoRotate = rotate
  }

  /** Draw one frame, advancing the camera flight if there is one. */
  private draw = (): void => {
    if (this.transition !== null) {
      const { from, to, start } = this.transition
      const t = Math.min((performance.now() - start) / TRANSITION_MS, 1)
      const eased = d3.easeCubicInOut(t)
      const direction = from.clone().normalize().lerp(to.clone().normalize(), eased).normalize()
      this.camera.position.copy(direction.multiplyScalar(this.distance))
      this.transition = t < 1 ? this.transition : null
    }
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
    this.frame = requestAnimationFrame(this.draw)
  }

  start(): void {
    if (this.frame === null) {
      this.frame = requestAnimationFrame(this.draw)
    }
  }

  stop(): void {
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame)
      this.frame = null
    }
  }

  private disposePoints(): void {
    if (this.points !== null) {
      this.scene.remove(this.points)
      this.points.geometry.dispose()
      this.points.material.dispose()
      this.points = null
    }
  }

  dispose(): void {
    this.stop()
    this.disposePoints()
    this.controls.dispose()
    this.renderer.dispose()
  }
}
