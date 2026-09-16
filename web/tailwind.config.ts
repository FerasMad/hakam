import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        carbon: "#090B0C",
        charcoal: "#111416",
        raised: "#181C1E",
        chalk: "#F2F0E9",
        muted: "#A2A9A6",
        hairline: "#2C3335",
        pitch: "#31C795",
        copper: "#C49073",
        danger: "#D67272",
      },
      fontFamily: {
        sans: ["Inter Variable", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        arabic: ["IBM Plex Sans Arabic", "Tahoma", "Arial", "sans-serif"],
      },
      letterSpacing: {
        broadcast: "0.16em",
      },
      maxWidth: {
        canvas: "85rem",
      },
      boxShadow: {
        station: "0 28px 90px rgba(0, 0, 0, 0.34)",
      },
    },
  },
  plugins: [],
};

export default config;
