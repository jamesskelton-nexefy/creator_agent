/**
 * Image Generator Library
 * 
 * Wrapper for Replicate's google/nano-banana-pro model for AI image generation.
 * Provides eLearning-friendly presets and handles API integration.
 */

import Replicate from "replicate";

// ============================================================================
// TYPES
// ============================================================================

/** Supported aspect ratios from nano-banana-pro API */
export type AspectRatio = 
  | "1:1" 
  | "2:3" 
  | "3:2" 
  | "3:4" 
  | "4:3" 
  | "4:5" 
  | "5:4" 
  | "9:16" 
  | "16:9" 
  | "21:9"
  | "match_input_image";

/** Supported output formats */
export type OutputFormat = "png" | "jpg";

/** eLearning-specific presets */
export type ImagePreset = 
  | "banner"      // 21:9 - Course/module banners
  | "hero"        // 16:9 - Hero images, slides
  | "content"     // 16:9 - General content images  
  | "thumbnail"   // 3:2 - Card thumbnails
  | "square"      // 1:1 - Icons, avatars
  | "portrait"    // 3:4 - Portrait photos
  | "custom";     // User-specified ratio

/** Preset configuration mapping */
export interface PresetConfig {
  aspectRatio: AspectRatio;
  description: string;
  useCases: string[];
}

/** Image generation input parameters */
export interface GenerateImageInput {
  /** Text description of the image to generate */
  prompt: string;
  /** eLearning preset (determines aspect ratio) */
  preset?: ImagePreset;
  /** Override aspect ratio (used when preset is "custom") */
  aspectRatio?: AspectRatio;
  /** Output format - default "png" */
  outputFormat?: OutputFormat;
  /** Resolution - default "2K" */
  resolution?: string;
}

/** Result from image generation */
export interface GenerateImageResult {
  /** URL of the generated image (from Replicate) */
  url: string;
  /** Image as Buffer (downloaded from Replicate) */
  buffer: Buffer;
  /** Aspect ratio used */
  aspectRatio: AspectRatio;
  /** Output format */
  outputFormat: OutputFormat;
  /** The prompt used */
  prompt: string;
  /** Preset used (if any) */
  preset?: ImagePreset;
  /** Generation timestamp */
  generatedAt: string;
}

// ============================================================================
// PRESETS CONFIGURATION
// ============================================================================

/**
 * eLearning-specific image presets.
 * Maps preset names to aspect ratios optimized for different use cases.
 */
export const IMAGE_PRESETS: Record<ImagePreset, PresetConfig> = {
  banner: {
    aspectRatio: "21:9",
    description: "Wide banner format for course headers and module banners",
    useCases: ["Course banners", "Module headers", "Landing page heroes"],
  },
  hero: {
    aspectRatio: "16:9",
    description: "Standard widescreen format for hero images and slides",
    useCases: ["Hero images", "Presentation slides", "Video thumbnails"],
  },
  content: {
    aspectRatio: "16:9",
    description: "General purpose content images",
    useCases: ["Inline content", "Explainer images", "Diagrams"],
  },
  thumbnail: {
    aspectRatio: "3:2",
    description: "Compact format for card thumbnails and previews",
    useCases: ["Course cards", "Module previews", "Gallery thumbnails"],
  },
  square: {
    aspectRatio: "1:1",
    description: "Square format for icons and avatars",
    useCases: ["Profile avatars", "Icons", "Social media"],
  },
  portrait: {
    aspectRatio: "3:4",
    description: "Portrait orientation for character images",
    useCases: ["Character portraits", "Person photos", "Mobile-first content"],
  },
  custom: {
    aspectRatio: "16:9", // Default, will be overridden
    description: "Custom aspect ratio specified by user",
    useCases: ["Flexible sizing"],
  },
};

// ============================================================================
// IMAGE GENERATOR CLASS
// ============================================================================

/**
 * Image generator using Replicate's nano-banana-pro model.
 */
export class ImageGenerator {
  private replicate: Replicate;
  private model = "google/nano-banana-pro" as const;

  constructor(apiToken?: string) {
    this.replicate = new Replicate({
      auth: apiToken || process.env.REPLICATE_API_TOKEN,
    });
  }

  /**
   * Generate an image using nano-banana-pro.
   * 
   * @param input Generation parameters
   * @returns Generated image result with URL and buffer
   */
  async generateImage(input: GenerateImageInput): Promise<GenerateImageResult> {
    const {
      prompt,
      preset = "content",
      aspectRatio: customAspectRatio,
      outputFormat = "png",
      resolution = "2K",
    } = input;

    // Determine aspect ratio from preset or custom value
    const aspectRatio = preset === "custom" && customAspectRatio
      ? customAspectRatio
      : IMAGE_PRESETS[preset].aspectRatio;

    console.log(`[imageGenerator] Generating image with preset="${preset}", aspectRatio="${aspectRatio}"`);
    console.log(`[imageGenerator] Prompt: "${prompt.substring(0, 100)}${prompt.length > 100 ? '...' : ''}"`);

    try {
      // Call Replicate API
      const output = await this.replicate.run(this.model, {
        input: {
          prompt,
          resolution,
          image_input: [],
          aspect_ratio: aspectRatio,
          output_format: outputFormat,
          safety_filter_level: "block_only_high",
        },
      });

      // Get the URL from the output (async because FileOutput.url() returns a Promise<URL>)
      const imageUrl = await this.extractUrl(output);
      
      if (!imageUrl || typeof imageUrl !== 'string') {
        throw new Error(`No valid image URL returned from Replicate. Got: ${typeof imageUrl}`);
      }

      console.log(`[imageGenerator] Image generated successfully: ${imageUrl.substring(0, 80)}...`);

      // Download the image as a buffer
      const buffer = await this.downloadImage(imageUrl);

      console.log(`[imageGenerator] Image downloaded: ${buffer.length} bytes`);

      return {
        url: imageUrl,
        buffer,
        aspectRatio,
        outputFormat,
        prompt,
        preset: preset !== "custom" ? preset : undefined,
        generatedAt: new Date().toISOString(),
      };
    } catch (error) {
      console.error("[imageGenerator] Generation failed:", error);
      throw new Error(
        `Image generation failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }

  /**
   * Extract URL from Replicate output.
   * Output can be a FileOutput object, string URL, URL object, or array.
   * FileOutput.url() returns a Promise<URL> in newer Replicate SDK versions.
   */
  private async extractUrl(output: unknown): Promise<string | null> {
    // Handle FileOutput object with url() method (returns Promise<URL> in newer SDK)
    if (output && typeof output === "object" && "url" in output) {
      const urlValue = (output as any).url;
      if (typeof urlValue === "function") {
        const result = await urlValue();
        // Result may be a URL object - convert to string
        if (result instanceof URL) {
          return result.href;
        }
        if (typeof result === "string") {
          return result;
        }
        // Handle nested URL object
        if (result && typeof result === "object" && "href" in result) {
          return result.href;
        }
      }
      if (typeof urlValue === "string") {
        return urlValue;
      }
    }

    // Handle native URL object
    if (output instanceof URL) {
      return output.href;
    }

    // Handle direct string URL
    if (typeof output === "string") {
      return output;
    }

    // Handle array of outputs (take first)
    if (Array.isArray(output) && output.length > 0) {
      return this.extractUrl(output[0]);
    }

    // Handle object with href property (URL-like)
    if (output && typeof output === "object" && "href" in output) {
      const href = (output as any).href;
      if (typeof href === "string") {
        return href;
      }
    }

    // Handle object with toString that returns URL
    if (output && typeof output === "object" && typeof (output as any).toString === "function") {
      const str = (output as any).toString();
      if (str.startsWith("http")) {
        return str;
      }
    }

    return null;
  }

  /**
   * Download image from URL to Buffer.
   */
  private async downloadImage(url: string): Promise<Buffer> {
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to download image: ${response.status} ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  }

  /**
   * Get preset information.
   */
  getPresetInfo(preset: ImagePreset): PresetConfig {
    return IMAGE_PRESETS[preset];
  }

  /**
   * List all available presets.
   */
  listPresets(): Array<{ name: ImagePreset; config: PresetConfig }> {
    return Object.entries(IMAGE_PRESETS).map(([name, config]) => ({
      name: name as ImagePreset,
      config,
    }));
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

let imageGeneratorInstance: ImageGenerator | null = null;

/**
 * Get or create the image generator instance.
 */
export function getImageGenerator(): ImageGenerator {
  if (!imageGeneratorInstance) {
    imageGeneratorInstance = new ImageGenerator();
  }
  return imageGeneratorInstance;
}

/**
 * Generate an image using the default generator instance.
 * Convenience function for simple usage.
 */
export async function generateImage(input: GenerateImageInput): Promise<GenerateImageResult> {
  return getImageGenerator().generateImage(input);
}

export default ImageGenerator;


