using System.Windows;
using MStudio.Services.Interfaces;

namespace MStudio.App
{
    public partial class MainWindow : Window
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;

        public MainWindow(ISessionService sessionService, ITimelineService timelineService)
        {
            InitializeComponent();
            _sessionService = sessionService;
            _timelineService = timelineService;

            Title = "MStudio .NET - Phase 1 Foundation";
        }
    }
}
