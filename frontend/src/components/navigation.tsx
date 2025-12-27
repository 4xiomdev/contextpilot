"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  Globe,
  Search,
  FileText,
  Activity,
  FlaskConical,
  MessageSquare,
  Boxes,
  Settings,
  Database,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useWebSocket } from "@/lib/websocket";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: Home },
  { href: "/crawl", label: "Crawl Manager", icon: Globe },
  { href: "/data", label: "Data Manager", icon: Database },
  { href: "/search", label: "Search", icon: Search },
  { href: "/docs", label: "Normalized Docs", icon: FileText },
];

const toolItems = [
  { href: "/playground", label: "MCP Playground", icon: FlaskConical },
  { href: "/chat", label: "Ask ContextPilot", icon: MessageSquare },
  { href: "/vectors", label: "Vector Explorer", icon: Boxes },
];

const bottomItems = [
  { href: "/setup", label: "Setup", icon: Settings },
];

export function Navigation() {
  const pathname = usePathname();
  const { status, isConnected } = useWebSocket();

  const NavLink = ({ href, label, icon: Icon }: { href: string; label: string; icon: React.ComponentType<{ className?: string }> }) => {
    const isActive = pathname === href;
    return (
      <Link
        href={href}
        className={cn(
          "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
          isActive
            ? "bg-violet-500/10 text-violet-400 border border-violet-500/20"
            : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50"
        )}
      >
        <Icon className="w-4 h-4" />
        {label}
      </Link>
    );
  };

  return (
    <nav className="fixed left-0 top-0 h-screen w-64 bg-zinc-900/50 border-r border-zinc-800 p-4 backdrop-blur-sm flex flex-col">
      <div className="mb-6">
        <div className="flex items-center gap-3 px-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-tight">ContextPilot</h1>
            <p className="text-xs text-zinc-500">Control Panel</p>
          </div>
        </div>
      </div>

      {/* Main Navigation */}
      <div className="space-y-1">
        <p className="text-[10px] uppercase tracking-wider text-zinc-600 px-3 mb-2">Data</p>
        {navItems.map((item) => (
          <NavLink key={item.href} {...item} />
        ))}
      </div>

      {/* Tools Section */}
      <div className="space-y-1 mt-6">
        <p className="text-[10px] uppercase tracking-wider text-zinc-600 px-3 mb-2">Tools</p>
        {toolItems.map((item) => (
          <NavLink key={item.href} {...item} />
        ))}
      </div>

      {/* Bottom Section */}
      <div className="mt-auto space-y-1">
        {bottomItems.map((item) => (
          <NavLink key={item.href} {...item} />
        ))}
      </div>

      {/* Connection Status */}
      <div className="mt-4">
        <div className="px-3 py-2 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
          <p className="text-xs text-zinc-500">Real-time Connection</p>
          <div className="flex items-center gap-2 mt-1">
            <div
              className={cn(
                "w-2 h-2 rounded-full",
                isConnected
                  ? "bg-emerald-500 animate-pulse"
                  : status === "connecting"
                  ? "bg-amber-500 animate-pulse"
                  : "bg-zinc-500"
              )}
            />
            <span className="text-xs text-zinc-400">
              {isConnected ? "Connected" : status === "connecting" ? "Connecting..." : "Disconnected"}
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
}


