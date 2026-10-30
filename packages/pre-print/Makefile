TEX      = pdflatex
BIB      = bibtex
MAIN     = ms

# Figures are NOT generated here. The figure set is produced by the `experiments`
# package CLI, run from the repo root, which writes a PNG (docs site) and a PDF (paper)
# into assets/figures/. This Makefile only *syncs* (copies) the finished PDFs in here,
# so the manuscript -- and its submission tarball -- stay self-contained.
ASSETSDIR = ../../assets/figures
FIGURES   = paper-eta.pdf paper-ratios.pdf paper-checks.pdf paper-flare.pdf paper-milky-way.pdf paper-inversion.pdf paper-imig.pdf paper-window.pdf

.PHONY: all clean distclean sync-figures submission

all: sync-figures $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(MAIN).bib $(FIGURES)
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

# Copy the finished figure PDFs from assets/figures/ into this directory. Figures
# must already exist there; regenerate them from the repo root with the experiments
# CLI, never from this folder.
sync-figures:
	$(if $(FIGURES),cp $(addprefix $(ASSETSDIR)/,$(FIGURES)) .)

OUTDIR   = /tmp/jr-hz-mono-age
STAGEDIR = $(OUTDIR)/submission
TARFILE  = $(OUTDIR)/jr-hz-mono-age-preprint.tar

# AAS policy (https://journals.aas.org/manuscript-preparation/#linenumber) asks for
# line numbers on the manuscript. The submission target injects aastex701's
# `linenumbers` option into a *staged* copy of $(MAIN).tex and compiles that, so the
# everyday build stays clean. Ships the prebuilt .bbl, so two pdflatex passes suffice.
submission: all
	@rm -rf $(STAGEDIR)
	@mkdir -p $(STAGEDIR)
	@sed -E 's/(\\documentclass\[[^]]*)\](\{aastex701\})/\1,linenumbers]\2/' $(MAIN).tex > $(STAGEDIR)/$(MAIN).tex
	@cp $(MAIN).bib $(MAIN).bbl $(STAGEDIR)/
	@$(if $(FIGURES),cp $(addprefix $(ASSETSDIR)/,$(FIGURES)) $(STAGEDIR)/)
	@cd $(STAGEDIR) && $(TEX) -interaction=nonstopmode -halt-on-error $(MAIN) && $(TEX) -interaction=nonstopmode -halt-on-error $(MAIN)
	@COPYFILE_DISABLE=1 tar cf $(TARFILE) -C $(STAGEDIR) $(MAIN).tex $(MAIN).bbl $(MAIN).pdf $(FIGURES)
	@echo "Created $(TARFILE) (line-numbered submission)"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf $(FIGURES)
