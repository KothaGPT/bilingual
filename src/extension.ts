import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

// ---- Activation ------------------------------------------------------------
export function activate(context: vscode.ExtensionContext) {
    // Simple hello world command
    const helloCmd = vscode.commands.registerCommand('warpStyleExtension.helloWorld', () => {
        vscode.window.showInformationMessage('Warp‑Style Extension is active!');
    });
    context.subscriptions.push(helloCmd);

    // Start a dummy language server (placeholder for future agents)
    const serverModule = context.asAbsolutePath('out/server.js');
    const serverOptions: ServerOptions = {
        run: { module: serverModule, transport: vscode.LanguageClient.TransportKind.ipc },
        debug: { module: serverModule, transport: vscode.LanguageClient.TransportKind.ipc, options: { execArgv: ['--nolazy', '--inspect=6009'] } }
    };
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'typescript' }],
        synchronize: { fileEvents: vscode.workspace.createFileSystemWatcher('**/*.ts') }
    };
    const client = new LanguageClient('warpStyleLS', 'Warp Style Language Server', serverOptions, clientOptions);
    context.subscriptions.push(client.start());

    // Placeholder webview panel for future AI prompts
    const panelCmd = vscode.commands.registerCommand('warpStyleExtension.openPromptPanel', () => {
        const panel = vscode.window.createWebviewPanel(
            'warpPrompt',
            'Warp Prompt',
            vscode.ViewColumn.One,
            { enableScripts: true }
        );
        panel.webview.html = `<html><body><h2>Warp Prompt UI – coming soon</h2></body></html>`;
    });
    context.subscriptions.push(panelCmd);
}

// ---- Deactivation -----------------------------------------------------------
export function deactivate() { }
