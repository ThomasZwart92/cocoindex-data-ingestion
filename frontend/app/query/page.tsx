'use client'

import { useCallback, useMemo, useRef, useState } from 'react'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

const MODELS = [
  { group: 'OpenAI', models: ['gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-mini'] },
  { group: 'Gemini', models: ['gemini-2.5-pro', 'gemini-2.5-flash'] },
]

// Fixed list matching metadata departments
const DEPARTMENTS = ['Sales','Marketing','Engineering','Support','Finance','Supply Chain','Logistics']
const SECURITY = ['public', 'client', 'partner', 'employee', 'management']

export default function QueryPage() {
  const [query, setQuery] = useState('')
  const [model, setModel] = useState('gpt-4o-mini')
  const [departments, setDepartments] = useState<string[]>([])
  const [securityTiers, setSecurityTiers] = useState<string[]>([])
  const [streaming, setStreaming] = useState(false)
  const [answer, setAnswer] = useState('')
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const canSubmit = useMemo(() => query.trim().length > 0 && !streaming, [query, streaming])

  const handleStop = useCallback(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
    setStreaming(false)
  }, [])

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!canSubmit) return
    setError(null)
    setAnswer('')
    setStreaming(true)

    const body = {
      query,
      model,
      departments: departments.length ? departments : undefined,
      security_tiers: securityTiers.length ? securityTiers : undefined,
      top_k: 6,
    }

    const controller = new AbortController()
    controllerRef.current = controller

    try {
      const res = await fetch(`${API_BASE_URL}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      if (!res.ok || !res.body) {
        const msg = await res.text()
        throw new Error(msg || `Request failed: ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        setAnswer((prev) => prev + chunk)
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') return
      setError(err?.message || 'Streaming failed')
    } finally {
      setStreaming(false)
      controllerRef.current = null
    }
  }, [query, model, departments, securityTiers, canSubmit])

  return (
    <div className="py-6" style={{ maxWidth: '85%', margin: '0 auto' }}>
      <h2 className="font-semibold mb-4" style={{ fontSize: '20px', color: '#2C3E50' }}>Query</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-600 mb-1">Model</label>
            <select
              className="w-full border rounded px-2 py-2"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              {MODELS.map((grp) => (
                <optgroup key={grp.group} label={grp.group}>
                  {grp.models.map((m) => (
                    <option value={m} key={m}>{m}</option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-600 mb-1">Departments</label>
            <div className="border rounded px-2 py-2 max-h-40 overflow-auto">
              {DEPARTMENTS.map((d) => {
                const checked = departments.includes(d)
                return (
                  <label key={d} className="flex items-center gap-2 text-sm py-1">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        setDepartments((prev) => e.target.checked ? [...prev, d] : prev.filter(x => x !== d))
                      }}
                    />
                    <span>{d}</span>
                  </label>
                )
              })}
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-600 mb-1">Security Tiers</label>
            <div className="border rounded px-2 py-2 max-h-40 overflow-auto">
              {SECURITY.map((s) => {
                const checked = securityTiers.includes(s)
                return (
                  <label key={s} className="flex items-center gap-2 text-sm py-1">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        setSecurityTiers((prev) => e.target.checked ? [...prev, s] : prev.filter(x => x !== s))
                      }}
                    />
                    <span>{s}</span>
                  </label>
                )
              })}
            </div>
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-600 mb-1">Question</label>
          <textarea
            className="w-full border rounded px-3 py-2"
            rows={4}
            placeholder="Ask a question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={!canSubmit}
            className="px-4 py-2 rounded text-white"
            style={{ backgroundColor: canSubmit ? '#3498DB' : '#AEB6BF' }}
          >
            {streaming ? 'Streaming…' : 'Ask'}
          </button>
          {streaming && (
            <button
              type="button"
              onClick={handleStop}
              className="px-3 py-2 rounded border"
            >
              Stop
            </button>
          )}
          {(departments.length > 0 || securityTiers.length > 0) && (
            <span className="text-sm text-gray-600">
              Filters: {(departments.length ? departments.join(', ') : 'any')} / {(securityTiers.length ? securityTiers.join(', ') : 'any')}
            </span>
          )}
        </div>
      </form>

      <div className="mt-6">
        <label className="block text-sm text-gray-600 mb-1">Answer</label>
        <div className="whitespace-pre-wrap border rounded p-3 bg-white min-h-[140px]">
          {answer || (streaming ? '…' : 'No answer yet')}
        </div>
        {error && (
          <div className="mt-2 text-sm text-red-600">{error}</div>
        )}
      </div>
    </div>
  )
}
