using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    public partial class GraphViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;

        [ObservableProperty]
        private int _selectedMarkerIndex = -1;

        [ObservableProperty]
        private PointCollection _pointsX = new();

        [ObservableProperty]
        private PointCollection _pointsY = new();

        [ObservableProperty]
        private PointCollection _pointsZ = new();

        [ObservableProperty]
        private double _viewWidth = 800;

        [ObservableProperty]
        private double _viewHeight = 100;

        [ObservableProperty]
        private double _cursorPosition = 0;

        public GraphViewModel(ISessionService sessionService, ITimelineService timelineService)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;

            _sessionService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ISessionService.CurrentMotion))
                {
                    UpdatePoints();
                }
            };

            _timelineService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ITimelineService.CurrentFrame))
                {
                    UpdateCursor();
                }
            };

            PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(SelectedMarkerIndex))
                {
                    UpdatePoints();
                }
            };
        }

        public void UpdatePoints()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || SelectedMarkerIndex < 0 || motion.Markers.MarkerCount <= SelectedMarkerIndex)
            {
                PointsX = new();
                PointsY = new();
                PointsZ = new();
                return;
            }

            var px = new PointCollection();
            var py = new PointCollection();
            var pz = new PointCollection();

            int frameCount = motion.Markers.FrameCount;
            if (frameCount <= 0) return;

            // Simple scaling logic to fit the view
            double stepX = ViewWidth / Math.Max(1, frameCount - 1);
            float minY = -0.5f, maxY = 2.0f; // Typical marker height range in meters
            double rangeY = maxY - minY;

            for (int f = 0; f < frameCount; f++)
            {
                var pos = motion.Markers.GetPosition(SelectedMarkerIndex, f);
                double x = f * stepX;

                if (!float.IsNaN(pos.X) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                {
                    // Map meter values to pixel space (Y is inverted in WPF)
                    px.Add(new Point(x, ViewHeight * (1 - (pos.X - minY) / rangeY)));
                    py.Add(new Point(x, ViewHeight * (1 - (pos.Y - minY) / rangeY)));
                    pz.Add(new Point(x, ViewHeight * (1 - (pos.Z - minY) / rangeY)));
                }
            }

            PointsX = px;
            PointsY = py;
            PointsZ = pz;
            UpdateCursor();
        }

        private void UpdateCursor()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers.FrameCount <= 0)
            {
                CursorPosition = 0;
                return;
            }

            double stepX = ViewWidth / Math.Max(1, motion.Markers.FrameCount - 1);
            CursorPosition = _timelineService.CurrentFrame * stepX;
        }
    }
}
