using System.ComponentModel;

namespace MStudio.Core.Interfaces
{
    /// <summary>
    /// Service interface for managing visualization settings.
    /// 
    /// Clean Architecture: Core layer interface that defines the contract
    /// for visualization settings without any platform-specific dependencies.
    /// </summary>
    public interface IVisualizationSettingsService : INotifyPropertyChanged
    {
        /// <summary>
        /// Marker radius in meters. Range: 0.005 ~ 0.05
        /// </summary>
        float MarkerSize { get; set; }

        /// <summary>
        /// Marker color as RGBA (0-1 range)
        /// </summary>
        (float R, float G, float B, float A) MarkerColor { get; set; }

        /// <summary>
        /// Marker opacity. Range: 0.1 ~ 1.0
        /// </summary>
        float MarkerOpacity { get; set; }

        /// <summary>
        /// Selected marker highlight color as RGBA (0-1 range)
        /// </summary>
        (float R, float G, float B, float A) SelectedMarkerColor { get; set; }

        /// <summary>
        /// Bone/skeleton line thickness. Range: 0.5 ~ 5.0
        /// </summary>
        double BoneThickness { get; set; }

        /// <summary>
        /// Bone color as RGBA (0-1 range)
        /// </summary>
        (float R, float G, float B, float A) BoneColor { get; set; }

        /// <summary>
        /// Bone opacity. Range: 0.1 ~ 1.0
        /// </summary>
        float BoneOpacity { get; set; }

        /// <summary>
        /// Whether marker names are displayed
        /// </summary>
        bool ShowMarkerNames { get; set; }

        /// <summary>
        /// Currently selected color scheme name
        /// </summary>
        string CurrentColorScheme { get; set; }

        /// <summary>
        /// Resets all settings to their default values
        /// </summary>
        void ResetToDefaults();

        /// <summary>
        /// Applies a predefined color scheme
        /// </summary>
        void ApplyColorScheme(string schemeName);

        /// <summary>
        /// Gets available color scheme names
        /// </summary>
        string[] GetAvailableColorSchemes();
    }
}

