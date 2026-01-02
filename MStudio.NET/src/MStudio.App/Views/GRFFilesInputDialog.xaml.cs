using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using Microsoft.Win32;

namespace MStudio.App.Views
{
    public partial class GRFFilesInputDialog : Window
    {
        public string? BodyKinematicsCsvPath { get; private set; }
        public bool Confirmed { get; private set; } = false;

        // Required columns for GRF estimation
        private static readonly string[] RequiredColumns = { "time", "COM_x", "COM_y", "COM_z" };

        public GRFFilesInputDialog()
        {
            InitializeComponent();
        }

        private void Header_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        private void BrowseBodyKin_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select BodyKinematics CSV Output",
                Filter = "CSV Files (*.csv)|*.csv|All Files (*.*)|*.*",
                CheckFileExists = true
            };

            if (dialog.ShowDialog() == true)
            {
                BodyKinCsvPath.Text = dialog.FileName;
                ValidateFile(dialog.FileName);
            }
        }

        private void ValidateFile(string filePath)
        {
            ValidationPanel.Visibility = Visibility.Collapsed;
            ErrorPanel.Visibility = Visibility.Collapsed;
            ConfirmButton.IsEnabled = false;
            ConfirmButton.Opacity = 0.5;

            if (!File.Exists(filePath))
            {
                ShowError("File does not exist.");
                return;
            }

            try
            {
                // Read first line to check headers
                var firstLine = File.ReadLines(filePath).FirstOrDefault();
                if (string.IsNullOrEmpty(firstLine))
                {
                    ShowError("File is empty.");
                    return;
                }

                // OpenSim BodyKinematics uses tab separator and may have "# times" header
                // Support both comma and tab separators
                char separator = firstLine.Contains('\t') ? '\t' : ',';
                var headers = firstLine.Split(separator)
                    .Select(h => h.Trim().TrimStart('#').Trim().ToLowerInvariant())
                    .ToList();

                // Check for required columns - accept "times" or "time" for time column
                var missingColumns = new System.Collections.Generic.List<string>();
                
                // Check time column (accept "time" or "times")
                if (!headers.Contains("time") && !headers.Contains("times"))
                {
                    missingColumns.Add("time/times");
                }
                
                // Check COM columns
                if (!headers.Contains("com_x")) missingColumns.Add("COM_x");
                if (!headers.Contains("com_y")) missingColumns.Add("COM_y");
                if (!headers.Contains("com_z")) missingColumns.Add("COM_z");

                if (missingColumns.Any())
                {
                    ShowError($"Missing required columns: {string.Join(", ", missingColumns)}");
                    return;
                }

                // Count data rows
                var lineCount = File.ReadLines(filePath).Count() - 1; // Exclude header

                ShowSuccess(filePath, lineCount);
                BodyKinematicsCsvPath = filePath;
                ConfirmButton.IsEnabled = true;
                ConfirmButton.Opacity = 1.0;
            }
            catch (System.Exception ex)
            {
                ShowError($"Error reading file: {ex.Message}");
            }
        }

        private void ShowSuccess(string filePath, int dataRows)
        {
            ValidationPanel.Visibility = Visibility.Visible;
            ValidationIcon.Text = "✓";
            ValidationIcon.Foreground = new System.Windows.Media.SolidColorBrush(
                (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString("#7ED957"));
            ValidationText.Text = "File validated successfully";
            ValidationText.Foreground = ValidationIcon.Foreground;
            ValidationDetails.Text = $"{Path.GetFileName(filePath)} - {dataRows} frames";
        }

        private void ShowError(string message)
        {
            ErrorPanel.Visibility = Visibility.Visible;
            ErrorText.Text = message;
        }

        private void ConfirmButton_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(BodyKinematicsCsvPath))
            {
                Confirmed = true;
                DialogResult = true;
                Close();
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
