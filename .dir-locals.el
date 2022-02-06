;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((python-mode
  . (
     (lsp-docker+-server-id . pyright)
     (lsp-docker+-docker-server-id . pyr-docker)
     (lsp-docker+-server-command . "pyright-langserver --stdio")
     (lsp-docker+-server-cmd-fn . lsp-docker+-exec-in-container)
     (lsp-docker+-priority . 10)
     (lsp-docker+-image-id . "ml_gpu_python")
     (lsp-docker+-container-name . "ml_gpu_python_1")
     (lsp-docker+-docker-options . "-u ${USER}")
     (lsp-docker+-path-mappings . (("${HOME}/work/" . "${HOME}/work/")))
     )))
