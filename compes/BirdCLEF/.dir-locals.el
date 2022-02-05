((python-mode
  . ((yapfify-executable . "docker run -i --rm docker_birdclef yapf")
     (lsp-docker+-server-id . pyright)
     (lsp-docker+-docker-server-id . pyls-docker)
     (lsp-docker+-server-command . "pyright-langserver --stdio")
     (lsp-docker+-image-id . "ml_gpu_jupyter")
     (lsp-docker+-container-name . "py-lsp-docker")
     (lsp-docker+-path-mappings . (("${HOME}/work/" . "${HOME}/work/"))))))
