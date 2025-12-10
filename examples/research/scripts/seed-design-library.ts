/**
 * Seed Design Library Script
 * 
 * One-time script to generate and populate the design library with:
 * - Image type/style examples (AI-generated via Replicate)
 * - Color palette swatches
 * - Component variant previews
 * - Animation style examples
 * - Typography pairing previews
 * 
 * Usage:
 *   npx tsx scripts/seed-design-library.ts
 * 
 * Environment variables required:
 *   - REPLICATE_API_TOKEN
 *   - SUPABASE_URL (or uses local default)
 *   - SUPABASE_SERVICE_ROLE_KEY
 */

import "dotenv/config";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { ImageGenerator } from "../lib/imageGenerator";

// ============================================================================
// CONFIGURATION
// ============================================================================

const SUPABASE_URL = process.env.SUPABASE_URL || "http://127.0.0.1:54321";
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const STORAGE_BUCKET = "design-library";

// Delay between image generations to avoid rate limiting
const GENERATION_DELAY_MS = 2000;

// ============================================================================
// ASSET DEFINITIONS
// ============================================================================

/**
 * Image Types - First stage of selection (multi-select)
 */
const IMAGE_TYPES = [
  {
    name: "Photos",
    description: "Professional photography for realistic workplace scenarios",
    config: { type: "photo" },
    prompt: "Professional stock photo collage showing diverse workplace scenarios, business meeting, office environment, natural lighting, high quality photography",
  },
  {
    name: "Illustrations",
    description: "Clean vector-style graphics for conceptual content",
    config: { type: "illustration" },
    prompt: "Collection of flat vector illustrations, modern minimalist style, vibrant colors, abstract business concepts, clean design",
  },
  {
    name: "3D Graphics",
    description: "Modern 3D renders for engaging visuals",
    config: { type: "3d" },
    prompt: "Isometric 3D illustration collection, modern tech style, soft shadows, colorful geometric shapes, clean render",
  },
  {
    name: "Icons & Line Art",
    description: "Simple outline icons for UI elements and diagrams",
    config: { type: "icon" },
    prompt: "Collection of minimal line art icons, thin strokes, simple geometric shapes, monochrome outline style, clean vector design",
  },
];

/**
 * Image Styles - Second stage (single-select per type)
 */
const IMAGE_STYLES = {
  photo: [
    {
      name: "Corporate Professional",
      description: "Clean, modern business environments with professional attire",
      config: { style: "corporate", lighting: "natural", mood: "professional" },
      prompt: "Professional corporate photograph, modern office interior, business people in suits collaborating, natural daylight, clean contemporary design, high-end photography",
    },
    {
      name: "Casual Lifestyle",
      description: "Relaxed, approachable workplace settings",
      config: { style: "casual", lighting: "warm", mood: "friendly" },
      prompt: "Casual workplace photograph, creative office space, diverse team in casual attire, warm natural lighting, relaxed atmosphere, modern coworking environment",
    },
    {
      name: "Healthcare Clinical",
      description: "Trust-inspiring medical and healthcare settings",
      config: { style: "healthcare", lighting: "bright", mood: "trustworthy" },
      prompt: "Healthcare professional photograph, clean clinical environment, medical staff in white coats, bright sterile lighting, modern hospital setting, reassuring atmosphere",
    },
    {
      name: "Industrial Technical",
      description: "Hands-on technical and manufacturing environments",
      config: { style: "industrial", lighting: "practical", mood: "competent" },
      prompt: "Industrial workplace photograph, manufacturing facility, workers with safety equipment, practical lighting, technical environment, professional safety gear",
    },
  ],
  illustration: [
    {
      name: "Flat Minimalist",
      description: "Simple shapes with limited color palette",
      config: { style: "flat", complexity: "simple", colors: "limited" },
      prompt: "Flat vector illustration, minimalist design, simple geometric shapes, limited color palette of 3-4 colors, clean lines, modern corporate style",
    },
    {
      name: "Hand-drawn Sketch",
      description: "Organic, hand-crafted feel with subtle textures",
      config: { style: "sketch", complexity: "medium", colors: "muted" },
      prompt: "Hand-drawn sketch style illustration, organic lines, subtle paper texture, pencil and watercolor effect, warm muted colors, artistic imperfect feel",
    },
    {
      name: "Isometric Technical",
      description: "3D isometric perspective with clean geometry",
      config: { style: "isometric", complexity: "detailed", colors: "vibrant" },
      prompt: "Isometric vector illustration, technical precision, 3D perspective, clean geometric shapes, vibrant gradient colors, modern tech aesthetic",
    },
    {
      name: "Gradient Abstract",
      description: "Flowing gradients with abstract shapes",
      config: { style: "gradient", complexity: "simple", colors: "gradient" },
      prompt: "Abstract gradient illustration, flowing organic shapes, smooth color transitions, purple to blue gradient, modern abstract art style, soft edges",
    },
  ],
  "3d": [
    {
      name: "Soft Isometric",
      description: "Friendly 3D with soft shadows and rounded edges",
      config: { style: "soft", perspective: "isometric", lighting: "soft" },
      prompt: "Soft 3D isometric render, rounded edges, pastel colors, gentle shadows, friendly cartoon style, clean background, modern 3D illustration",
    },
    {
      name: "Realistic Render",
      description: "Photorealistic 3D with detailed materials",
      config: { style: "realistic", perspective: "perspective", lighting: "studio" },
      prompt: "Photorealistic 3D render, detailed materials and textures, studio lighting, professional product visualization, high quality render",
    },
    {
      name: "Low-poly Stylized",
      description: "Geometric low-poly aesthetic",
      config: { style: "lowpoly", perspective: "isometric", lighting: "flat" },
      prompt: "Low-poly 3D illustration, geometric faceted style, flat shading, vibrant colors, modern game art aesthetic, clean triangular shapes",
    },
  ],
  icon: [
    {
      name: "Outlined Minimal",
      description: "Thin line icons with consistent stroke weight",
      config: { style: "outlined", strokeWeight: "thin", filled: false },
      prompt: "Minimal outline icon set, thin consistent stroke weight, simple geometric shapes, line art style, clean vector icons, monochrome black on white",
    },
    {
      name: "Filled Solid",
      description: "Solid filled icons for high contrast",
      config: { style: "filled", strokeWeight: "none", filled: true },
      prompt: "Solid filled icon set, high contrast shapes, no outlines, simple silhouette style, bold black icons, clean minimal design",
    },
    {
      name: "Duotone",
      description: "Two-tone icons with depth",
      config: { style: "duotone", strokeWeight: "medium", filled: "partial" },
      prompt: "Duotone icon set, two-color design, blue and light blue tones, partial fills for depth, modern app icon style, clean vector graphics",
    },
  ],
};

/**
 * Color Palettes with full color specifications
 */
const COLOR_PALETTES = [
  {
    name: "Corporate Blue",
    description: "Professional blue tones for business training",
    config: {
      primary: "#1e40af",
      secondary: "#64748b",
      accent: "#0ea5e9",
      background: "#f8fafc",
      text: "#1e293b",
      gradientStart: "#1e40af",
      gradientEnd: "#3b82f6",
    },
    tags: ["corporate", "professional", "trust"],
  },
  {
    name: "Healthcare Teal",
    description: "Calming teal palette for medical content",
    config: {
      primary: "#0d9488",
      secondary: "#6b7280",
      accent: "#10b981",
      background: "#f0fdfa",
      text: "#134e4a",
      gradientStart: "#0d9488",
      gradientEnd: "#14b8a6",
    },
    tags: ["healthcare", "medical", "calm"],
  },
  {
    name: "Tech Violet",
    description: "Modern purple tones for technical content",
    config: {
      primary: "#7c3aed",
      secondary: "#4b5563",
      accent: "#8b5cf6",
      background: "#faf5ff",
      text: "#1f2937",
      gradientStart: "#7c3aed",
      gradientEnd: "#a78bfa",
    },
    tags: ["tech", "modern", "innovation"],
  },
  {
    name: "Hospitality Warm",
    description: "Welcoming warm tones for service training",
    config: {
      primary: "#ea580c",
      secondary: "#78716c",
      accent: "#f59e0b",
      background: "#fffbeb",
      text: "#292524",
      gradientStart: "#ea580c",
      gradientEnd: "#fb923c",
    },
    tags: ["hospitality", "warm", "friendly"],
  },
  {
    name: "Education Red",
    description: "Classic academic reds for educational content",
    config: {
      primary: "#dc2626",
      secondary: "#6b7280",
      accent: "#2563eb",
      background: "#fef2f2",
      text: "#1f2937",
      gradientStart: "#dc2626",
      gradientEnd: "#ef4444",
    },
    tags: ["education", "academic", "traditional"],
  },
  {
    name: "Modern Green",
    description: "Fresh green palette for engaging content",
    config: {
      primary: "#059669",
      secondary: "#64748b",
      accent: "#f472b6",
      background: "#ecfdf5",
      text: "#111827",
      gradientStart: "#059669",
      gradientEnd: "#10b981",
    },
    tags: ["modern", "fresh", "growth"],
  },
  {
    name: "Dark Mode",
    description: "Dark theme for reduced eye strain",
    config: {
      primary: "#3b82f6",
      secondary: "#9ca3af",
      accent: "#f59e0b",
      background: "#111827",
      text: "#f9fafb",
      gradientStart: "#1e3a8a",
      gradientEnd: "#3b82f6",
    },
    tags: ["dark", "night", "accessibility"],
  },
  {
    name: "Light Minimal",
    description: "Clean white theme with subtle accents",
    config: {
      primary: "#374151",
      secondary: "#9ca3af",
      accent: "#6366f1",
      background: "#ffffff",
      text: "#111827",
      gradientStart: "#f3f4f6",
      gradientEnd: "#ffffff",
    },
    tags: ["minimal", "clean", "light"],
  },
];

/**
 * Animation style configurations
 */
const ANIMATION_STYLES = [
  {
    name: "None",
    description: "No animations - static content only",
    config: { enabled: false },
    tags: ["accessibility", "performance"],
  },
  {
    name: "Subtle",
    description: "Gentle fade-ins for a professional feel",
    config: {
      enabled: true,
      style: "subtle",
      duration: "fast",
      entranceType: "fade",
      durationMs: 200,
    },
    tags: ["professional", "elegant"],
  },
  {
    name: "Professional",
    description: "Smooth slide-up animations",
    config: {
      enabled: true,
      style: "moderate",
      duration: "normal",
      entranceType: "slide",
      durationMs: 400,
    },
    tags: ["business", "polished"],
  },
  {
    name: "Dynamic",
    description: "Bold scale animations for engaging content",
    config: {
      enabled: true,
      style: "dynamic",
      duration: "normal",
      entranceType: "scale",
      durationMs: 500,
    },
    tags: ["engaging", "modern", "energetic"],
  },
];

/**
 * Typography pairings
 */
const TYPOGRAPHY_PAIRINGS = [
  {
    name: "Inter + Open Sans",
    description: "Clean, modern sans-serif combination",
    config: {
      headingFont: "Inter",
      bodyFont: "Open Sans",
      style: "formal",
      headingWeight: 700,
      bodyWeight: 400,
    },
    tags: ["corporate", "modern", "clean"],
  },
  {
    name: "Poppins + Nunito",
    description: "Friendly, approachable pairing",
    config: {
      headingFont: "Poppins",
      bodyFont: "Nunito",
      style: "friendly",
      headingWeight: 600,
      bodyWeight: 400,
    },
    tags: ["friendly", "modern", "approachable"],
  },
  {
    name: "Merriweather + Georgia",
    description: "Classic serif combination for academic content",
    config: {
      headingFont: "Merriweather",
      bodyFont: "Georgia",
      style: "formal",
      headingWeight: 700,
      bodyWeight: 400,
    },
    tags: ["academic", "traditional", "elegant"],
  },
  {
    name: "Playfair Display + Lora",
    description: "Elegant serif pairing for premium content",
    config: {
      headingFont: "Playfair Display",
      bodyFont: "Lora",
      style: "formal",
      headingWeight: 700,
      bodyWeight: 400,
    },
    tags: ["premium", "elegant", "hospitality"],
  },
  {
    name: "JetBrains Mono + Inter",
    description: "Technical monospace for code-focused content",
    config: {
      headingFont: "JetBrains Mono",
      bodyFont: "Inter",
      style: "technical",
      headingWeight: 600,
      bodyWeight: 400,
    },
    tags: ["technical", "code", "developer"],
  },
  {
    name: "Lato + Source Sans Pro",
    description: "Healthcare-friendly professional pairing",
    config: {
      headingFont: "Lato",
      bodyFont: "Source Sans Pro",
      style: "formal",
      headingWeight: 700,
      bodyWeight: 400,
    },
    tags: ["healthcare", "professional", "clean"],
  },
  {
    name: "Montserrat + Roboto",
    description: "Modern geometric sans-serif duo",
    config: {
      headingFont: "Montserrat",
      bodyFont: "Roboto",
      style: "casual",
      headingWeight: 600,
      bodyWeight: 400,
    },
    tags: ["modern", "tech", "startup"],
  },
  {
    name: "DM Sans + DM Sans",
    description: "Versatile single-family solution",
    config: {
      headingFont: "DM Sans",
      bodyFont: "DM Sans",
      style: "casual",
      headingWeight: 700,
      bodyWeight: 400,
    },
    tags: ["versatile", "modern", "minimal"],
  },
];

/**
 * Component variant configurations
 */
const COMPONENT_VARIANTS = {
  TitleBlock: [
    { name: "Standard", variant: "standard", description: "Clean centered title" },
    { name: "Hero", variant: "hero", description: "Full-width hero with gradient" },
    { name: "Minimal", variant: "minimal", description: "Simple text-only header" },
    { name: "Gradient", variant: "gradient", description: "Bold gradient background" },
    { name: "Split", variant: "split", description: "Two-column layout with image" },
  ],
  TextBlock: [
    { name: "Standard", variant: "standard", description: "Regular paragraph text" },
    { name: "Callout", variant: "callout", description: "Highlighted callout box" },
    { name: "Quote", variant: "quote", description: "Styled blockquote" },
    { name: "Two Column", variant: "two-column", description: "Side-by-side columns" },
  ],
  QuestionBlock: [
    { name: "Standard", variant: "standard", description: "Traditional quiz format" },
    { name: "Card", variant: "card", description: "Options as clickable cards" },
    { name: "Gamified", variant: "gamified", description: "Game-style with points" },
  ],
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Generate a color palette swatch image
 */
function generatePaletteSwatchSVG(config: typeof COLOR_PALETTES[0]["config"]): Buffer {
  const svg = `
    <svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
      <rect width="300" height="200" fill="${config.background}"/>
      <rect x="10" y="10" width="80" height="80" rx="8" fill="${config.primary}"/>
      <rect x="100" y="10" width="80" height="80" rx="8" fill="${config.secondary}"/>
      <rect x="190" y="10" width="80" height="80" rx="8" fill="${config.accent}"/>
      <rect x="10" y="100" width="260" height="40" rx="8" fill="url(#gradient)"/>
      <rect x="10" y="150" width="260" height="40" rx="4" fill="${config.text}" opacity="0.1"/>
      <text x="20" y="175" font-family="system-ui" font-size="14" fill="${config.text}">Sample Text</text>
      <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:${config.gradientStart}"/>
          <stop offset="100%" style="stop-color:${config.gradientEnd}"/>
        </linearGradient>
      </defs>
    </svg>
  `.trim();
  return Buffer.from(svg);
}

/**
 * Generate a typography preview SVG
 */
function generateTypographySVG(config: typeof TYPOGRAPHY_PAIRINGS[0]["config"]): Buffer {
  const svg = `
    <svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
      <rect width="300" height="200" fill="#f8fafc"/>
      <text x="20" y="50" font-family="${config.headingFont}, system-ui" font-size="28" font-weight="${config.headingWeight}" fill="#1e293b">Heading</text>
      <text x="20" y="90" font-family="${config.headingFont}, system-ui" font-size="20" font-weight="${config.headingWeight}" fill="#334155">Subheading</text>
      <text x="20" y="130" font-family="${config.bodyFont}, system-ui" font-size="14" font-weight="${config.bodyWeight}" fill="#475569">Body text appears here in the</text>
      <text x="20" y="150" font-family="${config.bodyFont}, system-ui" font-size="14" font-weight="${config.bodyWeight}" fill="#475569">selected body font style.</text>
      <text x="20" y="180" font-family="system-ui" font-size="10" fill="#94a3b8">${config.headingFont} + ${config.bodyFont}</text>
    </svg>
  `.trim();
  return Buffer.from(svg);
}

/**
 * Generate animation preview placeholder (would be GIF in production)
 */
function generateAnimationPlaceholderSVG(name: string, config: any): Buffer {
  const colors = {
    None: "#94a3b8",
    Subtle: "#3b82f6",
    Professional: "#8b5cf6",
    Dynamic: "#f59e0b",
  };
  const color = colors[name as keyof typeof colors] || "#64748b";
  
  const svg = `
    <svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
      <rect width="300" height="200" fill="#1e293b"/>
      <circle cx="150" cy="80" r="30" fill="${color}" opacity="0.8"/>
      <rect x="50" y="130" width="200" height="20" rx="4" fill="${color}" opacity="0.6"/>
      <rect x="80" y="160" width="140" height="12" rx="3" fill="${color}" opacity="0.4"/>
      <text x="150" y="190" font-family="system-ui" font-size="12" fill="#94a3b8" text-anchor="middle">${name} Animation</text>
    </svg>
  `.trim();
  return Buffer.from(svg);
}

// ============================================================================
// MAIN SEEDING FUNCTIONS
// ============================================================================

async function seedImageTypes(
  supabase: SupabaseClient,
  generator: ImageGenerator
): Promise<void> {
  console.log("\n📷 Seeding Image Types...");
  
  for (let i = 0; i < IMAGE_TYPES.length; i++) {
    const item = IMAGE_TYPES[i];
    console.log(`  [${i + 1}/${IMAGE_TYPES.length}] Generating: ${item.name}`);
    
    try {
      // Generate image
      const result = await generator.generateImage({
        prompt: item.prompt,
        preset: "thumbnail",
      });
      
      // Upload to storage
      const fileName = `image-types/${item.config.type}.png`;
      const { error: uploadError } = await supabase.storage
        .from(STORAGE_BUCKET)
        .upload(fileName, result.buffer, {
          contentType: "image/png",
          upsert: true,
        });
      
      if (uploadError) {
        console.error(`    Upload error: ${uploadError.message}`);
        continue;
      }
      
      // Get public URL
      const { data: urlData } = supabase.storage
        .from(STORAGE_BUCKET)
        .getPublicUrl(fileName);
      
      // Insert into database
      const { error: insertError } = await supabase
        .from("design_library")
        .insert({
          category: "image_type",
          name: item.name,
          description: item.description,
          thumbnail_url: urlData.publicUrl,
          config: item.config,
          tags: [item.config.type],
          sort_order: i,
        });
      
      if (insertError) {
        console.error(`    Insert error: ${insertError.message}`);
      } else {
        console.log(`    ✓ ${item.name}`);
      }
      
      await sleep(GENERATION_DELAY_MS);
    } catch (error) {
      console.error(`    Error: ${error}`);
    }
  }
}

async function seedImageStyles(
  supabase: SupabaseClient,
  generator: ImageGenerator
): Promise<void> {
  console.log("\n🎨 Seeding Image Styles...");
  
  for (const [type, styles] of Object.entries(IMAGE_STYLES)) {
    console.log(`  Type: ${type}`);
    
    for (let i = 0; i < styles.length; i++) {
      const style = styles[i];
      console.log(`    [${i + 1}/${styles.length}] Generating: ${style.name}`);
      
      try {
        // Generate image
        const result = await generator.generateImage({
          prompt: style.prompt,
          preset: "thumbnail",
        });
        
        // Upload to storage
        const fileName = `image-styles/${type}-${style.config.style}.png`;
        const { error: uploadError } = await supabase.storage
          .from(STORAGE_BUCKET)
          .upload(fileName, result.buffer, {
            contentType: "image/png",
            upsert: true,
          });
        
        if (uploadError) {
          console.error(`      Upload error: ${uploadError.message}`);
          continue;
        }
        
        // Get public URL
        const { data: urlData } = supabase.storage
          .from(STORAGE_BUCKET)
          .getPublicUrl(fileName);
        
        // Insert into database
        const { error: insertError } = await supabase
          .from("design_library")
          .insert({
            category: "image_style",
            subcategory: type,
            name: style.name,
            description: style.description,
            thumbnail_url: urlData.publicUrl,
            config: style.config,
            tags: [type, style.config.style],
            sort_order: i,
          });
        
        if (insertError) {
          console.error(`      Insert error: ${insertError.message}`);
        } else {
          console.log(`      ✓ ${style.name}`);
        }
        
        await sleep(GENERATION_DELAY_MS);
      } catch (error) {
        console.error(`      Error: ${error}`);
      }
    }
  }
}

async function seedColorPalettes(supabase: SupabaseClient): Promise<void> {
  console.log("\n🌈 Seeding Color Palettes...");
  
  for (let i = 0; i < COLOR_PALETTES.length; i++) {
    const palette = COLOR_PALETTES[i];
    console.log(`  [${i + 1}/${COLOR_PALETTES.length}] Creating: ${palette.name}`);
    
    try {
      // Generate SVG swatch
      const svgBuffer = generatePaletteSwatchSVG(palette.config);
      
      // Upload to storage
      const fileName = `color-palettes/${palette.name.toLowerCase().replace(/\s+/g, "-")}.svg`;
      const { error: uploadError } = await supabase.storage
        .from(STORAGE_BUCKET)
        .upload(fileName, svgBuffer, {
          contentType: "image/svg+xml",
          upsert: true,
        });
      
      if (uploadError) {
        console.error(`    Upload error: ${uploadError.message}`);
        continue;
      }
      
      // Get public URL
      const { data: urlData } = supabase.storage
        .from(STORAGE_BUCKET)
        .getPublicUrl(fileName);
      
      // Insert into database
      const { error: insertError } = await supabase
        .from("design_library")
        .insert({
          category: "color_palette",
          name: palette.name,
          description: palette.description,
          thumbnail_url: urlData.publicUrl,
          config: palette.config,
          tags: palette.tags,
          sort_order: i,
        });
      
      if (insertError) {
        console.error(`    Insert error: ${insertError.message}`);
      } else {
        console.log(`    ✓ ${palette.name}`);
      }
    } catch (error) {
      console.error(`    Error: ${error}`);
    }
  }
}

async function seedAnimationStyles(supabase: SupabaseClient): Promise<void> {
  console.log("\n✨ Seeding Animation Styles...");
  
  for (let i = 0; i < ANIMATION_STYLES.length; i++) {
    const animation = ANIMATION_STYLES[i];
    console.log(`  [${i + 1}/${ANIMATION_STYLES.length}] Creating: ${animation.name}`);
    
    try {
      // Generate placeholder SVG (would be GIF in production)
      const svgBuffer = generateAnimationPlaceholderSVG(animation.name, animation.config);
      
      // Upload to storage
      const fileName = `animation-styles/${animation.name.toLowerCase().replace(/\s+/g, "-")}.svg`;
      const { error: uploadError } = await supabase.storage
        .from(STORAGE_BUCKET)
        .upload(fileName, svgBuffer, {
          contentType: "image/svg+xml",
          upsert: true,
        });
      
      if (uploadError) {
        console.error(`    Upload error: ${uploadError.message}`);
        continue;
      }
      
      // Get public URL
      const { data: urlData } = supabase.storage
        .from(STORAGE_BUCKET)
        .getPublicUrl(fileName);
      
      // Insert into database
      const { error: insertError } = await supabase
        .from("design_library")
        .insert({
          category: "animation_style",
          name: animation.name,
          description: animation.description,
          thumbnail_url: urlData.publicUrl,
          config: animation.config,
          tags: animation.tags,
          sort_order: i,
        });
      
      if (insertError) {
        console.error(`    Insert error: ${insertError.message}`);
      } else {
        console.log(`    ✓ ${animation.name}`);
      }
    } catch (error) {
      console.error(`    Error: ${error}`);
    }
  }
}

async function seedTypographyPairings(supabase: SupabaseClient): Promise<void> {
  console.log("\n📝 Seeding Typography Pairings...");
  
  for (let i = 0; i < TYPOGRAPHY_PAIRINGS.length; i++) {
    const typography = TYPOGRAPHY_PAIRINGS[i];
    console.log(`  [${i + 1}/${TYPOGRAPHY_PAIRINGS.length}] Creating: ${typography.name}`);
    
    try {
      // Generate SVG preview
      const svgBuffer = generateTypographySVG(typography.config);
      
      // Upload to storage
      const fileName = `typography-pairings/${typography.name.toLowerCase().replace(/\s+/g, "-").replace(/\+/g, "-")}.svg`;
      const { error: uploadError } = await supabase.storage
        .from(STORAGE_BUCKET)
        .upload(fileName, svgBuffer, {
          contentType: "image/svg+xml",
          upsert: true,
        });
      
      if (uploadError) {
        console.error(`    Upload error: ${uploadError.message}`);
        continue;
      }
      
      // Get public URL
      const { data: urlData } = supabase.storage
        .from(STORAGE_BUCKET)
        .getPublicUrl(fileName);
      
      // Insert into database
      const { error: insertError } = await supabase
        .from("design_library")
        .insert({
          category: "typography",
          name: typography.name,
          description: typography.description,
          thumbnail_url: urlData.publicUrl,
          config: typography.config,
          tags: typography.tags,
          sort_order: i,
        });
      
      if (insertError) {
        console.error(`    Insert error: ${insertError.message}`);
      } else {
        console.log(`    ✓ ${typography.name}`);
      }
    } catch (error) {
      console.error(`    Error: ${error}`);
    }
  }
}

async function seedComponentVariants(
  supabase: SupabaseClient,
  generator: ImageGenerator
): Promise<void> {
  console.log("\n🧱 Seeding Component Variants...");
  
  for (const [baseType, variants] of Object.entries(COMPONENT_VARIANTS)) {
    console.log(`  Component: ${baseType}`);
    
    for (let i = 0; i < variants.length; i++) {
      const variant = variants[i];
      console.log(`    [${i + 1}/${variants.length}] Generating: ${variant.name}`);
      
      try {
        // Generate preview image
        const prompt = `UI component preview, ${baseType} ${variant.variant} style, e-learning slide design, clean modern interface, ${variant.description}, white background, professional design mockup`;
        
        const result = await generator.generateImage({
          prompt,
          preset: "content",
        });
        
        // Upload to storage
        const fileName = `component-variants/${baseType.toLowerCase()}-${variant.variant}.png`;
        const { error: uploadError } = await supabase.storage
          .from(STORAGE_BUCKET)
          .upload(fileName, result.buffer, {
            contentType: "image/png",
            upsert: true,
          });
        
        if (uploadError) {
          console.error(`      Upload error: ${uploadError.message}`);
          continue;
        }
        
        // Get public URL
        const { data: urlData } = supabase.storage
          .from(STORAGE_BUCKET)
          .getPublicUrl(fileName);
        
        // Insert into database
        const { error: insertError } = await supabase
          .from("design_library")
          .insert({
            category: "component_variant",
            subcategory: baseType,
            name: variant.name,
            description: variant.description,
            thumbnail_url: urlData.publicUrl,
            config: { baseType, variant: variant.variant },
            tags: [baseType.toLowerCase(), variant.variant],
            sort_order: i,
          });
        
        if (insertError) {
          console.error(`      Insert error: ${insertError.message}`);
        } else {
          console.log(`      ✓ ${variant.name}`);
        }
        
        await sleep(GENERATION_DELAY_MS);
      } catch (error) {
        console.error(`      Error: ${error}`);
      }
    }
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function main(): Promise<void> {
  console.log("🎨 Design Library Seeder");
  console.log("========================\n");
  
  // Validate environment
  if (!SUPABASE_SERVICE_KEY) {
    console.error("Error: SUPABASE_SERVICE_ROLE_KEY is required");
    process.exit(1);
  }
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("Error: REPLICATE_API_TOKEN is required for image generation");
    process.exit(1);
  }
  
  console.log(`Supabase URL: ${SUPABASE_URL}`);
  console.log(`Storage Bucket: ${STORAGE_BUCKET}\n`);
  
  // Initialize clients
  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
    auth: { persistSession: false },
  });
  const generator = new ImageGenerator();
  
  // Seed all categories
  try {
    // Clean up existing data first
    console.log("🧹 Clearing existing design library data...");
    const { error: deleteError } = await supabase
      .from("design_library")
      .delete()
      .neq("id", "00000000-0000-0000-0000-000000000000"); // Delete all rows
    
    if (deleteError) {
      console.error(`  Warning: Could not clear existing data: ${deleteError.message}`);
    } else {
      console.log("  ✓ Existing data cleared");
    }
    
    // SVG-based (no AI generation needed)
    await seedColorPalettes(supabase);
    await seedAnimationStyles(supabase);
    await seedTypographyPairings(supabase);
    
    // AI-generated images
    await seedImageTypes(supabase, generator);
    await seedImageStyles(supabase, generator);
    await seedComponentVariants(supabase, generator);
    
    console.log("\n✅ Design library seeding complete!");
    
    // Print summary
    const { data: summary } = await supabase
      .from("design_library")
      .select("category")
      .eq("is_active", true);
    
    if (summary) {
      const counts = summary.reduce((acc, item) => {
        acc[item.category] = (acc[item.category] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      console.log("\n📊 Summary:");
      for (const [category, count] of Object.entries(counts)) {
        console.log(`  ${category}: ${count} items`);
      }
      console.log(`  Total: ${summary.length} items`);
    }
  } catch (error) {
    console.error("\n❌ Seeding failed:", error);
    process.exit(1);
  }
}

main();

