using System.Windows;
using System.Windows.Input;
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

        private void LabelsListBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            // Intercept Space and Arrow keys to allow Window InputBindings to handle them
            if (e.Key == Key.Space || e.Key == Key.Left || e.Key == Key.Right)
            {
                e.Handled = true;
                
                // Manually invoke the command based on key
                if (DataContext is ViewModels.MainViewModel vm)
                {
                    if (e.Key == Key.Space)
                        vm.TogglePlayCommand.Execute(null);
                    else if (e.Key == Key.Left)
                        vm.StepBackwardCommand.Execute(null);
                    else if (e.Key == Key.Right)
                        vm.StepForwardCommand.Execute(null);
                }
            }
        }
    }
}
