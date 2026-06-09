import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "var(--color-canvas)",
        surface: "var(--color-surface)",
        "surface-muted": "var(--color-surface-muted)",
        primary: "var(--color-primary)",
        "primary-soft": "var(--color-primary-soft)",
        text: "var(--color-text)",
        "text-muted": "var(--color-text-muted)",
        "text-subtle": "var(--color-text-subtle)",
        border: "var(--color-border)",
        "border-soft": "var(--color-border-soft)",
        success: "var(--color-success)",
        "success-soft": "var(--color-success-soft)",
        warning: "var(--color-warning)",
        "warning-soft": "var(--color-warning-soft)",
        danger: "var(--color-danger)",
        "danger-soft": "var(--color-danger-soft)",
      },
      spacing: {
        1: "var(--space-1)",
        2: "var(--space-2)",
        3: "var(--space-3)",
        4: "var(--space-4)",
        5: "var(--space-5)",
        6: "var(--space-6)",
        8: "var(--space-8)",
      },
      fontSize: {
        xs: ["var(--font-xs)", { lineHeight: "1.4" }],
        sm: ["var(--font-sm)", { lineHeight: "1.45" }],
        md: ["var(--font-md)", { lineHeight: "1.5" }],
        base: ["var(--font-base)", { lineHeight: "1.5" }],
        lg: ["var(--font-lg)", { lineHeight: "1.3" }],
        xl: ["var(--font-xl)", { lineHeight: "1.25" }],
        "2xl": ["var(--font-2xl)", { lineHeight: "1.2" }],
      },
      borderRadius: {
        xs: "var(--radius-xs)",
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
        pill: "var(--radius-pill)",
      },
      boxShadow: {
        card: "var(--shadow-card)",
        resting: "var(--shadow-card-resting)",
        pop: "var(--shadow-pop)",
        control: "var(--shadow-control)",
      },
    },
  },
  plugins: [],
} satisfies Config;
