using System;
using System.Numerics;

namespace MStudio.Core.Models
{
    /// <summary>
    /// Configuration for marker visual properties.
    /// 
    /// Clean Architecture: Core layer model that defines marker appearance settings.
    /// This model is platform-agnostic and can be used across different rendering implementations.
    /// </summary>
    public class MarkerVisualConfig
    {
        /// <summary>
        /// Marker size (radius in meters). Range: 0.005 ~ 0.05
        /// </summary>
        public float Size { get; set; } = 0.012f;

        /// <summary>
        /// Normal marker color (RGBA, 0-1 range)
        /// </summary>
        public Vector4 ColorNormal { get; set; } = new Vector4(0.2f, 0.8f, 0.3f, 1.0f); // Green

        /// <summary>
        /// Selected marker color (RGBA, 0-1 range)
        /// </summary>
        public Vector4 ColorSelected { get; set; } = new Vector4(1.0f, 0.9f, 0.4f, 1.0f); // Yellow

        /// <summary>
        /// Marker opacity. Range: 0.1 ~ 1.0
        /// </summary>
        public float Opacity { get; set; } = 1.0f;

        /// <summary>
        /// Creates a copy of this configuration
        /// </summary>
        public MarkerVisualConfig Clone()
        {
            return new MarkerVisualConfig
            {
                Size = this.Size,
                ColorNormal = this.ColorNormal,
                ColorSelected = this.ColorSelected,
                Opacity = this.Opacity
            };
        }
    }

    /// <summary>
    /// Configuration for skeleton/bone visual properties.
    /// </summary>
    public class SkeletonVisualConfig
    {
        /// <summary>
        /// Line width in pixels. Range: 0.5 ~ 5.0
        /// </summary>
        public float LineWidth { get; set; } = 1.5f;

        /// <summary>
        /// Bone color (RGBA, 0-1 range)
        /// </summary>
        public Vector4 Color { get; set; } = new Vector4(0.78f, 0.78f, 0.78f, 0.7f); // Light gray

        /// <summary>
        /// Bone opacity. Range: 0.1 ~ 1.0
        /// </summary>
        public float Opacity { get; set; } = 0.8f;

        /// <summary>
        /// Creates a copy of this configuration
        /// </summary>
        public SkeletonVisualConfig Clone()
        {
            return new SkeletonVisualConfig
            {
                LineWidth = this.LineWidth,
                Color = this.Color,
                Opacity = this.Opacity
            };
        }
    }

    /// <summary>
    /// Predefined color schemes for visualization
    /// </summary>
    public static class ColorSchemes
    {
        public static readonly (string Name, Vector4 MarkerNormal, Vector4 MarkerSelected, Vector4 Bone)[] Presets = new[]
        {
            ("Default", new Vector4(0.2f, 0.8f, 0.3f, 1.0f), new Vector4(1.0f, 0.9f, 0.4f, 1.0f), new Vector4(0.78f, 0.78f, 0.78f, 0.7f)),
            ("Blue", new Vector4(0.4f, 0.6f, 1.0f, 1.0f), new Vector4(0.0f, 0.5f, 1.0f, 1.0f), new Vector4(0.5f, 0.7f, 1.0f, 0.7f)),
            ("Green", new Vector4(0.4f, 1.0f, 0.4f, 1.0f), new Vector4(0.0f, 1.0f, 0.0f, 1.0f), new Vector4(0.5f, 0.9f, 0.5f, 0.7f)),
            ("Warm", new Vector4(1.0f, 0.7f, 0.4f, 1.0f), new Vector4(1.0f, 0.6f, 0.0f, 1.0f), new Vector4(0.9f, 0.7f, 0.5f, 0.7f)),
            ("White", new Vector4(1.0f, 1.0f, 1.0f, 1.0f), new Vector4(1.0f, 0.9f, 0.4f, 1.0f), new Vector4(0.9f, 0.9f, 0.9f, 0.7f))
        };
    }
}
