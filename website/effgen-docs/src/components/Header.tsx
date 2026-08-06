import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { Sun, Moon, Menu, Code } from 'lucide-react';
import './Header.css';

interface HeaderProps {
  onMenuClick: () => void;
}

export default function Header({ onMenuClick }: HeaderProps) {
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();

  // Get current page title for breadcrumb
  const getPageTitle = () => {
    const path = location.pathname;
    if (path === '/' || path === '/introduction') return 'Introduction';
    if (path.startsWith('/')) {
      const page = path.slice(1); // Remove leading slash
      return page.charAt(0).toUpperCase() + page.slice(1).replace(/-/g, ' ');
    }
    return '';
  };

  const pageTitle = getPageTitle();

  return (
    <header className="header">
      <div className="header-left">
        <button className="menu-toggle" onClick={onMenuClick} aria-label="Toggle menu">
          <Menu size={22} />
        </button>
        <nav className="header-nav">
          <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>Home</NavLink>
          <NavLink to="/introduction" className={({ isActive }) => `nav-item ${location.pathname.startsWith('/') ? 'active' : ''}`}>Docs</NavLink>
          <NavLink to="/examples" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>Examples</NavLink>
          <NavLink to="/guides" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>Guides</NavLink>
          <NavLink to="/api-reference" className={({ isActive }) => `nav-item api-ref ${location.pathname === '/api-reference' ? 'active' : ''}`}>
            <Code size={16} />
            <span>API</span>
          </NavLink>
        </nav>

        {/* Breadcrumb indicator */}
        {pageTitle && location.pathname !== '/' && (
          <div className="header-breadcrumb">
            <span className="breadcrumb-separator">/</span>
            <span className="breadcrumb-current">{pageTitle}</span>
          </div>
        )}
      </div>

      <div className="header-right">
        <a
          href="https://github.com/ctrl-gaurav/effGen"
          target="_blank"
          rel="noopener noreferrer"
          className="header-link"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
          </svg>
          <span className="link-text">GitHub</span>
        </a>
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  );
}
