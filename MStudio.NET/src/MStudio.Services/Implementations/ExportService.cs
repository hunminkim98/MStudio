using System;
using System.IO;
using System.Threading.Tasks;
using MStudio.Core.Models;
using MStudio.Core.Writers;
using MStudio.Services.Interfaces;

namespace MStudio.Services.Implementations
{
    /// <summary>
    /// Service for exporting motion data to TRC and C3D file formats.
    /// Uses IDialogService for platform-independent file dialogs.
    /// </summary>
    public class ExportService : IExportService
    {
        private readonly IDialogService _dialogService;
        
        private const string FileFilter = "TRC files (*.trc)|*.trc|C3D files (*.c3d)|*.c3d|All files (*.*)|*.*";
        
        public ExportService(IDialogService dialogService)
        {
            _dialogService = dialogService;
        }
        
        /// <summary>
        /// Saves the trial data to its original file path (overwrite).
        /// </summary>
        public async Task<bool> SaveAsync(Trial trial)
        {
            if (trial?.MotionData == null)
            {
                _dialogService.ShowError("No data to save.", "Save Error");
                return false;
            }
            
            var filePath = trial.FilePath;
            if (string.IsNullOrEmpty(filePath))
            {
                // If no original path, fall back to SaveAs
                return await SaveAsAsync(trial);
            }
            
            try
            {
                await WriteToFileAsync(filePath, trial.MotionData);
                _dialogService.ShowInfo($"Data saved to {filePath}", "Save Successful");
                return true;
            }
            catch (Exception ex)
            {
                _dialogService.ShowError($"Failed to save file: {ex.Message}", "Save Error");
                return false;
            }
        }
        
        /// <summary>
        /// Opens a save dialog and saves the trial data to the selected path.
        /// </summary>
        public async Task<bool> SaveAsAsync(Trial trial)
        {
            if (trial?.MotionData == null)
            {
                _dialogService.ShowError("No data to save.", "Save Error");
                return false;
            }
            
            // Get default filename from trial name
            var defaultFileName = !string.IsNullOrEmpty(trial.Name) 
                ? trial.Name 
                : "untitled";
            
            var filePath = _dialogService.ShowSaveFileDialog(FileFilter, defaultFileName, "Save As");
            
            if (string.IsNullOrEmpty(filePath))
            {
                return false; // User cancelled
            }
            
            try
            {
                await WriteToFileAsync(filePath, trial.MotionData);
                _dialogService.ShowInfo($"Data saved to {filePath}", "Save Successful");
                return true;
            }
            catch (Exception ex)
            {
                _dialogService.ShowError($"Failed to save file: {ex.Message}", "Save Error");
                return false;
            }
        }
        
        private static async Task WriteToFileAsync(string filePath, MotionData motionData)
        {
            var extension = Path.GetExtension(filePath).ToLowerInvariant();
            
            switch (extension)
            {
                case ".trc":
                    await TrcFileWriter.WriteAsync(filePath, motionData);
                    break;
                    
                case ".c3d":
                    await C3dFileWriter.WriteAsync(filePath, motionData);
                    break;
                    
                default:
                    throw new NotSupportedException($"Unsupported file format: {extension}");
            }
        }
    }
}
