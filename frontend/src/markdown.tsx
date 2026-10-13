import React from "react";

function escapeHtml(s: string) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!));
}

export function Markdown({ text }: { text: string }) {
  const parts = text.split(/```/);
  return (
    <div className="md">
      {parts.map((part, i) => {
        if (i % 2 === 1) {
          const nl = part.indexOf("\n");
          const code = nl === -1 ? part : part.slice(nl + 1);
          return (
            <pre key={i}>
              <code>{code}</code>
            </pre>
          );
        }
        const html = escapeHtml(part)
          .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
          .replace(/`([^`]+)`/g, "<code>$1</code>")
          .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
        return <div key={i} dangerouslySetInnerHTML={{ __html: html.replace(/\n/g, "<br/>") }} />;
      })}
    </div>
  );
}
