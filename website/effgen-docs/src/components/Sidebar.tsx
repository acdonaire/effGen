import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Book,
  Rocket,
  Settings,
  Bot,
  Brain,
  Wrench,
  Database,
  FileText,
  GitBranch,
  Globe,
  Zap,
  Cog,
  Code,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Search,
  X,
  Shield,
  Users,
  CheckCircle,
  Activity,
  Server,
  Cpu,
  Layers,
  Plug,
  Image,
  Lock,
  Cloud,
  ShieldCheck,
} from 'lucide-react';
import './Sidebar.css';
import { usePyPIVersion } from '../hooks/usePyPIVersion';

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const navSections: NavSection[] = [
  {
    title: 'Getting Started',
    items: [
      { label: 'Docs Home', path: '/home', icon: <BookOpen size={18} /> },
      { label: 'Introduction', path: '/introduction', icon: <Book size={18} /> },
      { label: 'Installation', path: '/installation', icon: <Settings size={18} /> },
      { label: 'Quick Start', path: '/quickstart', icon: <Rocket size={18} /> },
      { label: 'Release Notes', path: '/releases', icon: <Activity size={18} /> },
    ],
  },
  {
    title: 'Core Concepts',
    items: [
      { label: 'Agents', path: '/agents', icon: <Bot size={18} /> },
      { label: 'Models', path: '/models', icon: <Brain size={18} /> },
      { label: 'Tools', path: '/tools', icon: <Wrench size={18} /> },
      { label: 'Providers', path: '/providers', icon: <Server size={18} /> },
      { label: 'Native Provider Tools', path: '/native-provider-tools', icon: <Plug size={18} /> },
      { label: 'Memory', path: '/memory', icon: <Database size={18} /> },
      { label: 'Prompts', path: '/prompts', icon: <FileText size={18} /> },
      { label: 'Multimodal', path: '/multimodal', icon: <Image size={18} /> },
      { label: 'Hardware', path: '/hardware', icon: <Cpu size={18} /> },
    ],
  },
  {
    title: 'Observability & Reliability',
    items: [
      { label: 'Observability', path: '/observability', icon: <Activity size={18} /> },
      { label: 'Reliability', path: '/reliability', icon: <ShieldCheck size={18} /> },
    ],
  },
  {
    title: 'Safety',
    items: [
      { label: 'Guardrails', path: '/guardrails', icon: <Shield size={18} /> },
      { label: 'Human-in-the-Loop', path: '/human-loop', icon: <Users size={18} /> },
      { label: 'Security', path: '/security', icon: <Lock size={18} /> },
    ],
  },
  {
    title: 'RAG & Knowledge',
    items: [
      { label: 'RAG Pipeline', path: '/rag', icon: <BookOpen size={18} /> },
      { label: 'Domains', path: '/domains', icon: <Layers size={18} /> },
    ],
  },
  {
    title: 'Orchestration',
    items: [
      { label: 'Multi-Agent', path: '/multi-agent', icon: <GitBranch size={18} /> },
      { label: 'DAG Workflows', path: '/workflows', icon: <GitBranch size={18} /> },
      { label: 'Protocols', path: '/protocols', icon: <Globe size={18} /> },
      { label: 'Execution', path: '/execution', icon: <Zap size={18} /> },
      { label: 'Configuration', path: '/configuration', icon: <Cog size={18} /> },
    ],
  },
  {
    title: 'Evaluation & Debug',
    items: [
      { label: 'Evaluation', path: '/evaluation', icon: <CheckCircle size={18} /> },
      { label: 'Debugging', path: '/debug', icon: <Activity size={18} /> },
    ],
  },
  {
    title: 'Deployment',
    items: [
      { label: 'API Server', path: '/api-server', icon: <Server size={18} /> },
      { label: 'Deployment', path: '/deployment', icon: <Cloud size={18} /> },
      { label: 'Developer Experience', path: '/dx', icon: <Code size={18} /> },
      { label: 'Clients & SDKs', path: '/clients', icon: <Code size={18} /> },
      { label: 'Checkpointing', path: '/checkpointing', icon: <Database size={18} /> },
    ],
  },
  {
    title: 'Resources',
    items: [
      { label: 'API Reference', path: '/api-reference', icon: <Code size={18} /> },
      { label: 'Examples', path: '/examples', icon: <BookOpen size={18} /> },
      { label: 'Guides', path: '/guides', icon: <Book size={18} /> },
    ],
  },
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const location = useLocation();
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedSections, setExpandedSections] = useState<string[]>(
    navSections.map(s => s.title)
  );
  const pypiVersion = usePyPIVersion();

  const toggleSection = (title: string) => {
    setExpandedSections(prev =>
      prev.includes(title)
        ? prev.filter(t => t !== title)
        : [...prev, title]
    );
  };

  const filteredSections = navSections.map(section => ({
    ...section,
    items: section.items.filter(item =>
      item.label.toLowerCase().includes(searchQuery.toLowerCase())
    ),
  })).filter(section => section.items.length > 0);

  return (
    <>
      <div
        className={`sidebar-overlay ${isOpen ? 'active' : ''}`}
        onClick={onClose}
      />
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <a href="/" className="sidebar-logo" onClick={onClose}>
            <div className="logo-icon">
              <svg width="32" height="32" viewBox="0 0 40 40" fill="none">
                <path d="M20 2L36 11V29L20 38L4 29V11L20 2Z" stroke="url(#logoGrad)" strokeWidth="2.5" strokeLinejoin="round"/>
                <circle cx="20" cy="20" r="5" fill="url(#logoGrad)"/>
                <defs>
                  <linearGradient id="logoGrad" x1="4" y1="2" x2="36" y2="38">
                    <stop offset="0%" stopColor="#00ff88"/>
                    <stop offset="50%" stopColor="#00c96e"/>
                    <stop offset="100%" stopColor="#00e5ff"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span className="logo-text">effGen</span>
            <span className="version-badge">v{pypiVersion.loading ? "..." : pypiVersion.version}</span>
          </a>
          <button className="sidebar-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-search">
          <Search size={16} className="search-icon" />
          <input
            type="text"
            placeholder="Search docs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
          {searchQuery && (
            <button
              className="search-clear"
              onClick={() => setSearchQuery('')}
            >
              <X size={14} />
            </button>
          )}
        </div>

        <nav className="sidebar-nav">
          {filteredSections.map((section) => (
            <div key={section.title} className="nav-section">
              <button
                className="nav-section-title"
                onClick={() => toggleSection(section.title)}
              >
                <span>{section.title}</span>
                {expandedSections.includes(section.title) ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
              </button>
              {expandedSections.includes(section.title) && (
                <ul className="nav-list">
                  {section.items.map((item) => (
                    <li key={item.path}>
                      <NavLink
                        to={item.path}
                        className={({ isActive }) =>
                          `nav-link ${isActive ? 'active' : ''}`
                        }
                        onClick={onClose}
                      >
                        <span className="nav-icon">{item.icon}</span>
                        <span className="nav-label">{item.label}</span>
                      </NavLink>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <a
            href="https://github.com/ctrl-gaurav/effGen"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            <span>GitHub</span>
          </a>
        </div>
      </aside>
    </>
  );
}
