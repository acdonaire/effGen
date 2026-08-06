import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { useTheme } from '../context/ThemeContext';
import './MermaidDiagram.css';

interface MermaidDiagramProps {
  chart: string;
  title?: string;
}

export default function MermaidDiagram({ chart, title }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: theme === 'dark' ? 'dark' : 'default',
      themeVariables: theme === 'dark' ? {
        primaryColor: '#00c96e',
        primaryTextColor: '#e6efe9',
        primaryBorderColor: '#00ff88',
        lineColor: '#8fa89a',
        secondaryColor: '#1a221e',
        tertiaryColor: '#0f1512',
        background: '#0a0f0d',
        mainBkg: '#1a221e',
        secondBkg: '#0f1512',
        textColor: '#e6efe9',
        nodeBorder: '#00ff88',
        clusterBkg: '#151c18',
        clusterBorder: '#1e2e26',
        titleColor: '#e6efe9',
        edgeLabelBackground: '#1a221e',
      } : {
        primaryColor: '#00894a',
        primaryTextColor: '#1a1a1a',
        primaryBorderColor: '#00894a',
        lineColor: '#555555',
        secondaryColor: '#f7f8f8',
        tertiaryColor: '#f0f1f1',
        background: '#ffffff',
        mainBkg: '#ffffff',
        secondBkg: '#f7f8f8',
        textColor: '#1a1a1a',
        nodeBorder: '#00894a',
        clusterBkg: '#f0f1f1',
        clusterBorder: '#e2e2e2',
        titleColor: '#1a1a1a',
        edgeLabelBackground: '#ffffff',
      },
      flowchart: {
        curve: 'basis',
        padding: 20,
      },
      fontFamily: 'Inter, sans-serif',
    });

    const renderDiagram = async () => {
      try {
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        const { svg } = await mermaid.render(id, chart);
        setSvg(svg);
        setError(null);
      } catch (err) {
        console.error('Mermaid rendering error:', err);
        setError('Failed to render diagram');
      }
    };

    renderDiagram();
  }, [chart, theme]);

  return (
    <div className="mermaid-container">
      {title && <div className="mermaid-title">{title}</div>}
      <div className="mermaid-diagram" ref={containerRef}>
        {error ? (
          <div className="mermaid-error">{error}</div>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: svg }} />
        )}
      </div>
    </div>
  );
}
