import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/Layout';
import Home from './pages/Home';
import Introduction from './pages/Introduction';
import Installation from './pages/Installation';
import QuickStart from './pages/QuickStart';
import Agents from './pages/Agents';
import Models from './pages/Models';
import Tools from './pages/Tools';
import Memory from './pages/Memory';
import Prompts from './pages/Prompts';
import MultiAgent from './pages/MultiAgent';
import Protocols from './pages/Protocols';
import Execution from './pages/Execution';
import Configuration from './pages/Configuration';
import APIReference from './pages/APIReference';
import Examples from './pages/Examples';
import Guides from './pages/Guides';
import Guardrails from './pages/Guardrails';
import RAG from './pages/RAG';
import Workflows from './pages/Workflows';
import Evaluation from './pages/Evaluation';
import APIServer from './pages/APIServer';
import Checkpointing from './pages/Checkpointing';
import HumanLoop from './pages/HumanLoop';
import Debug from './pages/Debug';
import Domains from './pages/Domains';
import Clients from './pages/Clients';
import Hardware from './pages/Hardware';
import Releases from './pages/Releases';
import Providers from './pages/Providers';
import NativeProviderTools from './pages/NativeProviderTools';
import Multimodal from './pages/Multimodal';
import Observability from './pages/Observability';
import Reliability from './pages/Reliability';
import Security from './pages/Security';
import Deployment from './pages/Deployment';
import DeveloperExperience from './pages/DeveloperExperience';
import './styles/globals.css';

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter basename={import.meta.env.BASE_URL}>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/introduction" replace />} />

            {/* Getting Started */}
            <Route path="home" element={<Home />} />
            <Route path="introduction" element={<Introduction />} />
            <Route path="installation" element={<Installation />} />
            <Route path="quickstart" element={<QuickStart />} />
            <Route path="releases" element={<Releases />} />

            {/* Core Concepts */}
            <Route path="agents" element={<Agents />} />
            <Route path="models" element={<Models />} />
            <Route path="tools" element={<Tools />} />
            <Route path="providers" element={<Providers />} />
            <Route path="native-provider-tools" element={<NativeProviderTools />} />
            <Route path="memory" element={<Memory />} />
            <Route path="prompts" element={<Prompts />} />
            <Route path="multimodal" element={<Multimodal />} />

            {/* Advanced */}
            <Route path="multi-agent" element={<MultiAgent />} />
            <Route path="workflows" element={<Workflows />} />
            <Route path="protocols" element={<Protocols />} />
            <Route path="execution" element={<Execution />} />
            <Route path="configuration" element={<Configuration />} />
            <Route path="hardware" element={<Hardware />} />

            {/* Observability & Reliability */}
            <Route path="observability" element={<Observability />} />
            <Route path="reliability" element={<Reliability />} />

            {/* Safety */}
            <Route path="guardrails" element={<Guardrails />} />
            <Route path="human-loop" element={<HumanLoop />} />
            <Route path="security" element={<Security />} />

            {/* RAG & Domains */}
            <Route path="rag" element={<RAG />} />
            <Route path="domains" element={<Domains />} />

            {/* Evaluation & Debug */}
            <Route path="evaluation" element={<Evaluation />} />
            <Route path="debug" element={<Debug />} />

            {/* Deployment */}
            <Route path="api-server" element={<APIServer />} />
            <Route path="deployment" element={<Deployment />} />
            <Route path="dx" element={<DeveloperExperience />} />
            <Route path="clients" element={<Clients />} />
            <Route path="checkpointing" element={<Checkpointing />} />

            {/* Reference */}
            <Route path="api-reference" element={<APIReference />} />

            {/* Resources */}
            <Route path="examples" element={<Examples />} />
            <Route path="guides" element={<Guides />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
