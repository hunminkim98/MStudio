using System.Windows;
using Microsoft.Win32;
using MStudio.Services.Interfaces;

namespace MStudio.App.Services
{
    /// <summary>
    /// WPF implementation of IDialogService.
    /// 
    /// This class lives in the App layer because it depends on WPF-specific types:
    /// - Microsoft.Win32.OpenFileDialog
    /// - Microsoft.Win32.SaveFileDialog
    /// - System.Windows.MessageBox
    /// 
    /// The ViewModel depends only on IDialogService (from Services layer),
    /// maintaining Clean Architecture's dependency rule.
    /// 
    /// Benefits:
    /// - ViewModels are platform-agnostic and testable
    /// - Can swap implementations (WPF → Avalonia, Console, Mock)
    /// - Proper separation of concerns
    /// </summary>
    public class DialogService : IDialogService
    {
        public string? ShowOpenFileDialog(string filter, string? title = null)
        {
            var dialog = new OpenFileDialog
            {
                Filter = filter,
                Title = title ?? "Open File"
            };

            return dialog.ShowDialog() == true ? dialog.FileName : null;
        }

        public string? ShowSaveFileDialog(string filter, string? defaultFileName = null, string? title = null)
        {
            var dialog = new SaveFileDialog
            {
                Filter = filter,
                Title = title ?? "Save File",
                FileName = defaultFileName ?? string.Empty
            };

            return dialog.ShowDialog() == true ? dialog.FileName : null;
        }

        public void ShowInfo(string message, string? title = null)
        {
            MessageBox.Show(
                message,
                title ?? "Information",
                MessageBoxButton.OK,
                MessageBoxImage.Information);
        }

        public void ShowWarning(string message, string? title = null)
        {
            MessageBox.Show(
                message,
                title ?? "Warning",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
        }

        public void ShowError(string message, string? title = null)
        {
            MessageBox.Show(
                message,
                title ?? "Error",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }

        public bool ShowConfirmation(string message, string? title = null)
        {
            var result = MessageBox.Show(
                message,
                title ?? "Confirm",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question);

            return result == MessageBoxResult.Yes;
        }
    }
}
