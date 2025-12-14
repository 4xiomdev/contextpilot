import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable React Compiler
  reactCompiler: true,
  
  // Output as static site for Firebase Hosting
  output: "export",
  
  // Trailing slashes for Firebase
  trailingSlash: true,
  
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
  
  // Environment variables available to the browser
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    NEXT_PUBLIC_API_KEY: process.env.NEXT_PUBLIC_API_KEY || "",
  },
};

export default nextConfig;
