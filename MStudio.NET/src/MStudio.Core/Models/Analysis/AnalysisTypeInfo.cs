namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// Metadata for each analysis type including display information.
    /// </summary>
    public record AnalysisTypeInfo(
        AnalysisType Type,
        string DisplayName,
        string Description,
        string IconKey);
}
