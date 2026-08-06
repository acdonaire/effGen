import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import './PageNavbar.css';

interface NavItem {
  label: string;
  path: string;
  icon?: React.ReactNode;
}

interface PageNavbarProps {
  title: string;
  items: NavItem[];
  rightContent?: React.ReactNode;
}

export default function PageNavbar({ title, items, rightContent }: PageNavbarProps) {
  const location = useLocation();

  return (
    <div className="page-navbar">
      <div className="page-navbar-left">
        <h2 className="page-navbar-title">{title}</h2>
        <nav className="page-navbar-nav">
          {items.map((item, index) => (
            <NavLink
              key={index}
              to={item.path}
              className={({ isActive }) =>
                `page-nav-item ${isActive || location.pathname === item.path ? 'active' : ''}`
              }
            >
              {item.icon && <span className="page-nav-icon">{item.icon}</span>}
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
      {rightContent && (
        <div className="page-navbar-right">
          {rightContent}
        </div>
      )}
    </div>
  );
}
