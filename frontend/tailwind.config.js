/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        serif: ['Playfair Display', 'serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        finance: {
          bg: '#F8FAFC',       // 极淡的灰白背景，像高级纸张
          surface: '#FFFFFF',  // 纯白卡片
          ink: '#0F172A',      // 深邃的墨水蓝黑，替代纯黑
          subtle: '#64748B',   // 低调的灰色文字
          gold: '#D4AF37',     // 奢华金，用于强调
          alert: '#BE123C',    // 玫瑰红，用于高风险
          border: '#E2E8F0',   // 极细的边框色
        }
      },
      boxShadow: {
        'float': '0 20px 40px -10px rgba(0, 0, 0, 0.05)', // 悬浮感阴影
      }
    },
  },
  plugins: [],
}
