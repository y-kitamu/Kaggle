;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

(
 (nil
  . ((dir-local-docker-config-alists
      . (
         ((docker-image-id . "python_lsp")
          (docker-container-name . "py-lsp-docker")
          (lsp-docker-client-configs
           . ((:server-id pyright :docker-server-id pyright-docker :server-command "pyright-langserver --stdio"))))
         ))
     )
  )
 (python-mode . ((yapfify-executable . "docker run -i --rm ml_gpu_jupyter yapf")))
 )
