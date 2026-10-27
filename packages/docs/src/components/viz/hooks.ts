import { type RefObject, useEffect, useRef, useState } from 'react'

/** Width the server renders with, before the container has been measured. */
export const FALLBACK_WIDTH = 640

/**
 * The current width of a container, followed through a ResizeObserver. The
 * first render uses the fallback on both server and client, so hydration sees
 * the same markup; the measured width arrives in an effect.
 */
export function useElementWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(FALLBACK_WIDTH)
  useEffect(() => {
    const element = ref.current
    if (element === null) {
      return undefined
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry !== undefined) {
        setWidth(Math.max(Math.floor(entry.contentRect.width), 240))
      }
    })
    observer.observe(element)
    setWidth(Math.max(Math.floor(element.getBoundingClientRect().width), 240))
    return () => observer.disconnect()
  }, [])
  return [ref, width]
}

/** Whether an element is at least partly inside the viewport. */
export function useInView<T extends Element>(ref: RefObject<T | null>): boolean {
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const element = ref.current
    if (element === null) {
      return undefined
    }
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0]
        if (entry !== undefined) {
          setInView(entry.isIntersecting)
        }
      },
      { threshold: 0.05 }
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref])
  return inView
}

/** Whether the reader asked the system for reduced motion. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(query.matches)
    const listener = (event: MediaQueryListEvent): void => setReduced(event.matches)
    query.addEventListener('change', listener)
    return () => query.removeEventListener('change', listener)
  }, [])
  return reduced
}
