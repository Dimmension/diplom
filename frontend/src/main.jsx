import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import LlmPage from './LlmPage'
import './styles.css'

const normalizedPath = window.location.pathname.replace(/\/+$/, '') || '/'
const rootPage = normalizedPath === '/llm' ? <LlmPage /> : <App />

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {rootPage}
  </React.StrictMode>,
)
