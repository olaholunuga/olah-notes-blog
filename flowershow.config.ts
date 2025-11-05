import { defineConfig } from "flowershow";

export default defineConfig({
  site: {
    title: "Olaoluwa’s Energy Notes",
    description: "A digital garden on renewable energy, power systems, and ideas.",
    baseUrl: "https://notes.olaoluwaolunuga.me", // Change to your domain
  },

  content: {
    directory: "content",
    // Enable backlinks and tags
    backlinks: true,
    tags: true,
    wikiLinks: true,
  },

  theme: {
    colorScheme: "auto", // 'light' | 'dark' | 'auto'
    accentColor: "#3abf9e",
    showLastUpdated: true,
  },

  markdown: {
    // Enable GitHub-flavored markdown and callouts
    remarkPlugins: ["remark-gfm", "remark-callouts"],
    rehypePlugins: ["rehype-slug", "rehype-autolink-headings"],
  },

  navbar: {
    items: [
      { label: "Home", href: "/" },
      { label: "Projects", href: "/projects" }
    //   { label: "Ideas", href: "/ideas" },
    ],
  },

  footer: {
    text: "© 2025 Olaoluwa Olunuga — Built with Flowershow 🌸",
  },
});