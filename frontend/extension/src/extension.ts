import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import ignore from 'ignore';

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
    const disposable = vscode.commands.registerCommand('fraimwork.examineDependencies', async () => {
        vscode.window.showInformationMessage('Analyzing dependencies...');

        const files = await getFiles();
        console.log('Files: ', files);
        const graph: { [key: string]: string[] } = {};

        for (const file of files) {
            console.log('File: ', file.fsPath);
            const symbols: vscode.DocumentSymbol[] = await getSymbols(file);
            console.log('Symbols: ', symbols);

            for (const symbol of symbols) {
                const definitions = await resolveSymbolDefinition(file, symbol);
                console.log('Definitions: ', definitions);

                for (const def of definitions) {
                    if (!graph[file.path]) {
                        graph[file.path] = [];
                    }

                    graph[file.path].push(def.uri.fsPath);
                }
            }
        }

        visualizeDependencyGraph(graph);
    });

    context.subscriptions.push(disposable);
}


async function getFiles(): Promise<vscode.Uri[]> {
    const pattern = '**/*.{py}'; // Adjust this pattern based on the file types you're interested in
    const ignoredPattern = '**/node_modules/**'; // Adjust this pattern based on the file types you're interested in
    const files = await vscode.workspace.findFiles(pattern, ignoredPattern);
    return files;
}

async function getSymbols(uri: vscode.Uri): Promise<vscode.DocumentSymbol[]> {
    const document = await vscode.workspace.openTextDocument(uri);
    console.log('Document: ', document.getText());
    const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>('vscode.executeDocumentSymbolProvider', document.uri);
    return symbols || [];
}

async function resolveSymbolDefinition(uri: vscode.Uri, symbol: vscode.DocumentSymbol) {
    const position = new vscode.Position(symbol.selectionRange.start.line, symbol.selectionRange.start.character);
    const locations = await vscode.commands.executeCommand<vscode.Location[]>('vscode.executeDefinitionProvider', uri, position);
    
    return locations || [];
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
