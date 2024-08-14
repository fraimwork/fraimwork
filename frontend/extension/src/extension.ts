import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

const workspaceFolders = vscode.workspace.workspaceFolders;

if (workspaceFolders) {
    workspaceFolders.forEach((folder: { uri: { fsPath: any; }; }) => {
        console.log('Folder: ', folder.uri.fsPath);
    });
}

function getWebviewContent(context: vscode.ExtensionContext): string {
    const htmlFilePath = path.join(context.extensionPath, 'media', 'webview.html');
    const htmlContent = fs.readFileSync(htmlFilePath, 'utf8');
    return htmlContent;
}

export function activate(context: vscode.ExtensionContext) {
    let disposable = vscode.commands.registerCommand('extension.showDependencyGraphForAllFiles', async () => {
        vscode.window.showInformationMessage('Analyzing dependencies...');

        const files = await getAllFilesInWorkspace();
        const dependencyGraph: { [file: string]: string[] } = {};

        for (const fileUri of files) {
            const document = await vscode.workspace.openTextDocument(fileUri);
            const dependencies = await findDependenciesInDocument(document);
            dependencyGraph[fileUri.fsPath] = dependencies;
        }

        visualizeDependencyGraph(dependencyGraph);
    });

    context.subscriptions.push(disposable);
}

async function getAllFilesInWorkspace(): Promise<vscode.Uri[]> {
    const pattern = '**/*.{ts,js,py,dart}'; // Adjust this pattern based on the file types you're interested in
    return await vscode.workspace.findFiles(pattern);
}

async function analyzeDependenciesForAllFiles() {
    const files = await getAllFilesInWorkspace();
    const dependencyGraph: { [file: string]: string[] } = {};

    for (const fileUri of files) {
        const document = await vscode.workspace.openTextDocument(fileUri);
        const dependencies = await findDependenciesInDocument(document);
        dependencyGraph[fileUri.fsPath] = dependencies;
    }

    visualizeDependencyGraph(dependencyGraph);
}

async function findDependenciesInDocument(document: vscode.TextDocument): Promise<string[]> {
    const symbolPositions = findImportPositions(document);
    const dependencies: string[] = [];

    for (const position of symbolPositions) {
        const locations = await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeDefinitionProvider',
            document.uri,
            position
        );

        if (locations && locations.length > 0) {
            const targetFile = locations[0].uri.fsPath;
            dependencies.push(targetFile);
        }
    }

    return dependencies;
}

function findImportPositions(document: vscode.TextDocument): vscode.Position[] {
    const importPositions: vscode.Position[] = [];
    const text = document.getText();
    const lines = text.split('\n');

    lines.forEach((line, i) => {
        if (line.startsWith('import') || line.startsWith('from')) {
            importPositions.push(new vscode.Position(i, 0));
        }
    });

    return importPositions;
}

function visualizeDependencyGraph(dependencyGraph: { [file: string]: string[] }) {
    const panel = vscode.window.createWebviewPanel(
        'dependencyGraph',
        'Dependency Graph',
        vscode.ViewColumn.One,
        {}
    );

    const graphData = buildGraphData(dependencyGraph);
    panel.webview.html = renderGraph(graphData);
}

function buildGraphData(dependencyGraph: { [file: string]: string[] }) {
    const nodes = Object.keys(dependencyGraph).map(file => ({ id: file }));
    const edges = Object.keys(dependencyGraph).flatMap(file => {
        return dependencyGraph[file].map(dep => ({ from: file, to: dep }));
    });

    return { nodes, edges };
}

function renderGraph(graphData: any): string {
    // Use a visualization library like D3.js or render simple HTML/SVG.
    return `<html><body><pre>${JSON.stringify(graphData, null, 2)}</pre></body></html>`;
}

// This method is called when your extension is deactivated
export function deactivate() {}
