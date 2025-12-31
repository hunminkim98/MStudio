using System.Windows;
using MStudio.Core.Models.Analysis;

namespace MStudio.App.Views
{
    public partial class AnalysisSelectionWindow : Window
    {
        public AnalysisType? SelectedAnalysisType { get; private set; }

        public AnalysisSelectionWindow()
        {
            InitializeComponent();
            LoadAnalysisCards();
        }

        private void LoadAnalysisCards()
        {
            var analysisItems = new[]
            {
                new AnalysisCardItem
                {
                    Type = AnalysisType.Gait,
                    DisplayName = "Gait",
                    ShortDescription = "Analyze walking patterns",
                    IconData = "M8,2 L8,4 M8,5 L8,10 M8,10 L5,14 M8,10 L11,14 M8,6 L5,8 M8,6 L11,8",
                    IconColor = "#6C9BDB",
                    IconBackground = "#1E3A5F"
                },
                new AnalysisCardItem
                {
                    Type = AnalysisType.Running,
                    DisplayName = "Running",
                    ShortDescription = "Analyze running mechanics",
                    IconData = "M10,2 L10,4 M10,5 L8,8 M8,8 L4,10 M8,8 L11,12 M10,5 L13,6 M10,5 L7,4",
                    IconColor = "#7ED957",
                    IconBackground = "#1E4F2E"
                },
                new AnalysisCardItem
                {
                    Type = AnalysisType.Sprint,
                    DisplayName = "Sprint",
                    ShortDescription = "Analyze sprint performance",
                    IconData = "M12,3 L10,5 M10,5 L6,6 M10,5 L12,9 M6,6 L3,10 M6,6 L4,4 M14,5 L12,5",
                    IconColor = "#FFB347",
                    IconBackground = "#4F3A1E"
                },
                new AnalysisCardItem
                {
                    Type = AnalysisType.CounterMovementJump,
                    DisplayName = "Counter Movement Jump",
                    ShortDescription = "Analyze CMJ height & power",
                    IconData = "M8,14 L8,6 M8,6 L6,8 M8,6 L10,8 M5,14 L11,14 M6,3 L8,1 L10,3",
                    IconColor = "#E06C75",
                    IconBackground = "#4F1E2E"
                },
                new AnalysisCardItem
                {
                    Type = AnalysisType.BroadJump,
                    DisplayName = "Broad Jump",
                    ShortDescription = "Analyze standing broad jump",
                    IconData = "M2,12 L6,8 L10,8 L14,4 M14,4 L14,8 M14,4 L10,4",
                    IconColor = "#C678DD",
                    IconBackground = "#3F1E4F"
                },
                new AnalysisCardItem
                {
                    Type = AnalysisType.ChangeOfDirection,
                    DisplayName = "Change of Direction",
                    ShortDescription = "Analyze agility movements",
                    IconData = "M2,12 L8,6 L14,12 M8,6 L8,2",
                    IconColor = "#56B6C2",
                    IconBackground = "#1E3F4F"
                }
            };

            AnalysisCards.ItemsSource = analysisItems;
        }

        private void AnalysisCard_Click(object sender, RoutedEventArgs e)
        {
            if (sender is System.Windows.Controls.Button button && button.Tag is AnalysisType type)
            {
                SelectedAnalysisType = type;
                DialogResult = true;
                Close();
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }

        private void Header_MouseLeftButtonDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (e.LeftButton == System.Windows.Input.MouseButtonState.Pressed)
            {
                DragMove();
            }
        }
    }

    public class AnalysisCardItem
    {
        public AnalysisType Type { get; set; }
        public string DisplayName { get; set; } = string.Empty;
        public string ShortDescription { get; set; } = string.Empty;
        public string IconData { get; set; } = string.Empty;
        public string IconColor { get; set; } = "#FFFFFF";
        public string IconBackground { get; set; } = "#2A2A3E";
    }
}
