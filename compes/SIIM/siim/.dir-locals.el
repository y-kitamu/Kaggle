;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((rust-mode . ((lsp-docker+-server-id . rust-analyzer)
               (lsp-docker+-docker-server-id . rust-docker)
               (lsp-docker+-server-command . "rust-analyzer")
               (lsp-docker+-image-id . "rust_rust")
               (lsp-docker+-container-name . "rust-lsp-docker")
               (lsp-docker+-docker-options . "-u ${USER}")
               (lsp-docker+-path-mappings . (("/home/kitamura/work/" . "/home/kitamura/work/"))))))
