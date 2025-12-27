using System.ComponentModel;
using MStudio.Core.Interfaces;

namespace MStudio.Services
{
    /// <summary>
    /// Implementation of visualization settings service.
    /// 
    /// Clean Architecture: Services layer implementation that manages
    /// visualization settings with property change notifications.
    /// </summary>
    public class VisualizationSettingsService : IVisualizationSettingsService
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        // Default values
        private const float DefaultMarkerSize = 0.012f;
        private const float DefaultMarkerOpacity = 1.0f;
        private static readonly (float, float, float, float) DefaultMarkerColor = (0.2f, 0.8f, 0.3f, 1.0f); // Green
        private static readonly (float, float, float, float) DefaultSelectedMarkerColor = (1.0f, 0.9f, 0.4f, 1.0f); // Yellow
        private const double DefaultBoneThickness = 1.5;
        private const float DefaultBoneOpacity = 0.8f;
        private static readonly (float, float, float, float) DefaultBoneColor = (0.78f, 0.78f, 0.78f, 0.7f); // Light gray
        private const bool DefaultShowMarkerNames = false;
        private const string DefaultColorScheme = "Default";

        // Backing fields
        private float _markerSize = DefaultMarkerSize;
        private float _markerOpacity = DefaultMarkerOpacity;
        private (float R, float G, float B, float A) _markerColor = DefaultMarkerColor;
        private (float R, float G, float B, float A) _selectedMarkerColor = DefaultSelectedMarkerColor;
        private double _boneThickness = DefaultBoneThickness;
        private float _boneOpacity = DefaultBoneOpacity;
        private (float R, float G, float B, float A) _boneColor = DefaultBoneColor;
        private bool _showMarkerNames = DefaultShowMarkerNames;
        private string _currentColorScheme = DefaultColorScheme;

        // Color schemes (matching v1 Python implementation)
        private static readonly Dictionary<string, (
            (float, float, float, float) MarkerNormal,
            (float, float, float, float) MarkerSelected,
            (float, float, float, float) Bone)> ColorSchemes = new()
        {
            ["Default"] = ((0.2f, 0.8f, 0.3f, 1.0f), (1.0f, 0.9f, 0.4f, 1.0f), (0.78f, 0.78f, 0.78f, 0.7f)),
            ["White"] = ((1.0f, 1.0f, 1.0f, 1.0f), (1.0f, 0.9f, 0.4f, 1.0f), (0.7f, 0.7f, 0.7f, 0.8f)),
            ["Blue"] = ((0.7f, 0.8f, 1.0f, 1.0f), (0.0f, 0.5f, 1.0f, 1.0f), (0.5f, 0.7f, 1.0f, 0.7f)),
            ["Green"] = ((0.7f, 1.0f, 0.7f, 1.0f), (0.0f, 1.0f, 0.0f, 1.0f), (0.5f, 0.9f, 0.5f, 0.7f)),
            ["Warm"] = ((1.0f, 0.9f, 0.7f, 1.0f), (1.0f, 0.6f, 0.0f, 1.0f), (0.9f, 0.7f, 0.5f, 0.7f))
        };

        /// <inheritdoc/>
        public float MarkerSize
        {
            get => _markerSize;
            set
            {
                value = Math.Clamp(value, 0.005f, 0.05f);
                if (_markerSize != value)
                {
                    _markerSize = value;
                    OnPropertyChanged(nameof(MarkerSize));
                }
            }
        }

        /// <inheritdoc/>
        public float MarkerOpacity
        {
            get => _markerOpacity;
            set
            {
                value = Math.Clamp(value, 0.1f, 1.0f);
                if (_markerOpacity != value)
                {
                    _markerOpacity = value;
                    OnPropertyChanged(nameof(MarkerOpacity));
                }
            }
        }

        /// <inheritdoc/>
        public (float R, float G, float B, float A) MarkerColor
        {
            get => _markerColor;
            set
            {
                if (_markerColor != value)
                {
                    _markerColor = value;
                    OnPropertyChanged(nameof(MarkerColor));
                }
            }
        }

        /// <inheritdoc/>
        public (float R, float G, float B, float A) SelectedMarkerColor
        {
            get => _selectedMarkerColor;
            set
            {
                if (_selectedMarkerColor != value)
                {
                    _selectedMarkerColor = value;
                    OnPropertyChanged(nameof(SelectedMarkerColor));
                }
            }
        }

        /// <inheritdoc/>
        public double BoneThickness
        {
            get => _boneThickness;
            set
            {
                value = Math.Clamp(value, 0.5, 5.0);
                if (_boneThickness != value)
                {
                    _boneThickness = value;
                    OnPropertyChanged(nameof(BoneThickness));
                }
            }
        }

        /// <inheritdoc/>
        public float BoneOpacity
        {
            get => _boneOpacity;
            set
            {
                value = Math.Clamp(value, 0.1f, 1.0f);
                if (_boneOpacity != value)
                {
                    _boneOpacity = value;
                    OnPropertyChanged(nameof(BoneOpacity));
                }
            }
        }

        /// <inheritdoc/>
        public (float R, float G, float B, float A) BoneColor
        {
            get => _boneColor;
            set
            {
                if (_boneColor != value)
                {
                    _boneColor = value;
                    OnPropertyChanged(nameof(BoneColor));
                }
            }
        }

        /// <inheritdoc/>
        public bool ShowMarkerNames
        {
            get => _showMarkerNames;
            set
            {
                if (_showMarkerNames != value)
                {
                    _showMarkerNames = value;
                    OnPropertyChanged(nameof(ShowMarkerNames));
                }
            }
        }

        /// <inheritdoc/>
        public string CurrentColorScheme
        {
            get => _currentColorScheme;
            set
            {
                if (_currentColorScheme != value && ColorSchemes.ContainsKey(value))
                {
                    _currentColorScheme = value;
                    OnPropertyChanged(nameof(CurrentColorScheme));
                }
            }
        }

        /// <inheritdoc/>
        public void ResetToDefaults()
        {
            MarkerSize = DefaultMarkerSize;
            MarkerOpacity = DefaultMarkerOpacity;
            MarkerColor = DefaultMarkerColor;
            SelectedMarkerColor = DefaultSelectedMarkerColor;
            BoneThickness = DefaultBoneThickness;
            BoneOpacity = DefaultBoneOpacity;
            BoneColor = DefaultBoneColor;
            ShowMarkerNames = DefaultShowMarkerNames;
            CurrentColorScheme = DefaultColorScheme;
        }

        /// <inheritdoc/>
        public void ApplyColorScheme(string schemeName)
        {
            if (ColorSchemes.TryGetValue(schemeName, out var scheme))
            {
                MarkerColor = scheme.MarkerNormal;
                SelectedMarkerColor = scheme.MarkerSelected;
                BoneColor = scheme.Bone;
                CurrentColorScheme = schemeName;
            }
        }

        /// <inheritdoc/>
        public string[] GetAvailableColorSchemes()
        {
            return ColorSchemes.Keys.ToArray();
        }

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}

