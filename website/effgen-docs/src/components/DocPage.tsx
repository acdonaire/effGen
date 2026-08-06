import React, { ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import './DocPage.css';

interface Breadcrumb {
  label: string;
  path?: string;
}

interface DocPageProps {
  title: string;
  subtitle?: string;
  icon?: ReactNode;
  breadcrumbs?: Breadcrumb[];
  children: ReactNode;
}

export default function DocPage({
  title,
  subtitle,
  icon,
  breadcrumbs = [],
  children,
}: DocPageProps) {
  return (
    <div className="doc-page">
      {breadcrumbs.length > 0 && (
        <nav className="breadcrumbs">
          <Link to="/" className="breadcrumb-item">Home</Link>
          {breadcrumbs.map((crumb, index) => (
            <React.Fragment key={index}>
              <ChevronRight size={14} className="breadcrumb-separator" />
              {crumb.path ? (
                <Link to={crumb.path} className="breadcrumb-item">
                  {crumb.label}
                </Link>
              ) : (
                <span className="breadcrumb-item current">{crumb.label}</span>
              )}
            </React.Fragment>
          ))}
        </nav>
      )}

      <header className="doc-header">
        {icon && <div className="doc-icon">{icon}</div>}
        <h1 className="doc-title">{title}</h1>
        {subtitle && <p className="doc-subtitle">{subtitle}</p>}
      </header>

      <div className="doc-content">
        {children}
      </div>
    </div>
  );
}

// Reusable components for documentation
export function InfoBox({
  type = 'info',
  title,
  children,
}: {
  type?: 'info' | 'warning' | 'success' | 'error';
  title?: string;
  children: ReactNode;
}) {
  const icons = {
    info: '💡',
    warning: '⚠️',
    success: '✅',
    error: '❌',
  };

  return (
    <div className={`info-box ${type}`}>
      {title && (
        <div className="info-box-title">
          <span>{icons[type]}</span>
          <span>{title}</span>
        </div>
      )}
      <div className="info-box-content">{children}</div>
    </div>
  );
}

export function ApiTable({
  headers,
  rows,
}: {
  headers: string[];
  rows: (string | ReactNode)[][];
}) {
  return (
    <div className="api-table-container">
      <table className="api-table">
        <thead>
          <tr>
            {headers.map((header, i) => (
              <th key={i}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function QuickLinks({
  links,
}: {
  links: { icon: string; title: string; description: string; path: string }[];
}) {
  return (
    <div className="quick-links">
      {links.map((link, i) => (
        <Link key={i} to={link.path} className="quick-link-card">
          <div className="quick-link-icon">{link.icon}</div>
          <div className="quick-link-title">{link.title}</div>
          <div className="quick-link-desc">{link.description}</div>
        </Link>
      ))}
    </div>
  );
}

export function FeatureList({
  features,
}: {
  features: { icon: string; title: string; description: string }[];
}) {
  return (
    <div className="feature-list">
      {features.map((feature, i) => (
        <div key={i} className="feature-item">
          <span className="feature-icon">{feature.icon}</span>
          <div className="feature-content">
            <strong>{feature.title}</strong>
            <span>{feature.description}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
