using System.Windows.Controls;

namespace MStudio.App.Views
{
    /// <summary>
    /// GraphView - Time-series graph visualization for marker data.
    /// 
    /// Clean Architecture: All UI logic has been moved to Attached Behaviors:
    /// - SizeObserverBehavior: Observes ActualWidth/ActualHeight and binds to ViewModel
    /// - MouseSeekBehavior: Handles click/drag seeking to a frame position
    /// 
    /// This code-behind is now empty except for the required InitializeComponent call.
    /// All interactions are declaratively defined in XAML via behaviors.
    /// </summary>
    public partial class GraphView : UserControl
    {
        public GraphView()
        {
            InitializeComponent();
        }
    }
}


