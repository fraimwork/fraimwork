# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-23.11"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python311 
    pkgs.python311Packages.pip
    pkgs.nodejs_20
  ];

  # Sets environment variables in the workspace
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.python" # For Python development
      "donjayamanne.githistory" # For Git integration
      "esbenp.prettier-vscode" # For code formatting
      "dbaeumer.vscode-eslint" # For JavaScript linting
      "ms-python.debugpy"
      "PKief.material-icon-theme"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        web = {
          command = [ "cd backend/app && source .venv/bin/activate && python app.py" ];
          env = { PORT = "$PORT"; };
          manager = "web";
        };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        # install Python dependencies
        pip-install = "cd backend/app && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt";
        # install Node.js dependencies
        npm-install = "cd frontend && npm install";
      };
      # Runs when the workspace is (re)started
      onStart = {
        # install Python dependencies
        pip-install = "cd backend/app && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt";
        # install Node.js dependencies
        npm-install = "cd frontend && npm install";
      };
    };
  };
}
