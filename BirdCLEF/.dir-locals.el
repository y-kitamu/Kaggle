(
 (nil
  . ((dir-local-docker-config-alists
      . (
         ((docker-image-id . "docker_birdclef")
          (docker-container-name . "py-lsp-docker")
          (lsp-docker-client-configs
           . ((:server-id pyright :docker-server-id dpyright :server-command "pyright-langserver --stdio"))))
         ))
     )
  )
 (python-mode . ((yapfify-executable . "docker run -i --workdir=/home/kitamura --user=kitamura --rm docker_birdclef yapf")))
 )
