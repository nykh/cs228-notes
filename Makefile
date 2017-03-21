TEMPDIR := $(shell mktemp -d -t tmp)

publish:
	echo 'hmmm'
	cp -r ./_site/* $(TEMPDIR)
	cd $(TEMPDIR) && \
	ls -a  && \
	git init && \
	git add . && \
	git commit -m 'publish site' && \
	git remote add origin git@github.com:kuleshov/cs228-notes.git && \
	git push origin master:refs/heads/gh-pages --force

local:
	echo 'now publish locally'
	jekyll clean && jekyll build
	sudo rm -r /var/www/html/228 && sudo cp -r _site /var/www/html/228
