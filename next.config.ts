import path from "node:path";
import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  output: "export",
  basePath: process.env.PAGES_BASE_PATH || undefined,
  turbopack: {
    root: projectRoot,
  },
};

export default nextConfig;
