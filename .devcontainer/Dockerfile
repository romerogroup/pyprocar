FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install bash-completion and common tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion git ca-certificates curl locales \
    fzf zoxide ripgrep fd-find eza bat less man-db \
  && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 || true
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8


RUN update-alternatives --install /usr/local/bin/fd fd /usr/bin/fdfind 50

RUN apt-get update && apt-get install -y \
    git \
    curl \
    sudo \
    openssh-client \
    ca-certificates \
    ssh \
    && rm -rf /var/lib/apt/lists/*


# 2) Configure Bash globally: completion, fzf bindings, aliases, and your functions
RUN printf "%s\n" \
  "" \
  "# ---------- Bash completion ----------" \
  "if [ -f /usr/share/bash-completion/bash_completion ]; then" \
  "  . /usr/share/bash-completion/bash_completion" \
  "fi" \
  "" \
  "# ---------- fzf key bindings and completions ----------" \
  "[ -f /usr/share/doc/fzf/examples/key-bindings.bash ] && . /usr/share/doc/fzf/examples/key-bindings.bash" \
  "[ -f /usr/share/doc/fzf/examples/completion.bash ] && . /usr/share/doc/fzf/examples/completion.bash" \
  "" \
  "# ---------- Handy aliases ----------" \
  "alias ls='eza --group-directories-first --icons=auto --color=auto'" \
  "alias lsa='ls -la'" \
  "alias lt='eza -T -L 2 --group-directories-first --icons=auto'" \
  "alias lta='eza -Ta -L 2 --group-directories-first --icons=auto'" \
  "alias rg='rg --smart-case --hidden --glob \"!.git\"'" \
  "export MANPAGER='less -R'" \
  "" \
  "# ---------- ff: fuzzy file finder with preview ----------" \
  "ff() {" \
  "  local previewer='bat --style=plain --color=always --line-range=:200 {} || sed -n \"1,200p\" {}'" \
  "  fd --hidden --follow --exclude .git --type f | fzf --preview \"${previewer}\" --height 40% --layout=reverse --prompt='ff> '" \
  "}" \
  "" \
  "# ---------- zoxide (smarter cd) ----------" \
  "eval \"\$(zoxide init bash)\"" \
  "" \
  "# ---------- Archive helpers ----------" \
  "compress() {" \
  "  if [ \"\$#\" -lt 1 ]; then echo 'Usage: compress <archive.tar.gz> [paths...]'; return 2; fi" \
  "  local out=\"\$1\"; shift" \
  "  case \"\$out\" in *.tar.gz|*.tgz) ;; *) echo 'compress: output should end with .tar.gz or .tgz'; return 2;; esac" \
  "  [ \"\$#\" -eq 0 ] && set -- ." \
  "  mkdir -p \"\$(dirname -- \"\$out\")\" || return 1" \
  "  tar -czf \"\$out\" --exclude-vcs --exclude='*.pyc' --exclude='__pycache__' \"\$@\"" \
  "}" \
  "" \
  "decompress() {" \
  "  if [ \"\$#\" -ne 1 ]; then echo 'Usage: decompress <archive.tar.gz>'; return 2; fi" \
  "  local arc=\"\$1\"; [ -f \"\$arc\" ] || { echo \"decompress: file not found: \$arc\"; return 2; }" \
  "  case \"\$arc\" in *.tar.gz|*.tgz) tar -xzf \"\$arc\" ;; *) echo 'decompress: only .tar.gz/.tgz supported'; return 2;; esac" \
  "}" \
  "" \
  "# ---------- ImageMagick helper pick ----------" \
  "__imgmagick() { if command -v magick >/dev/null 2>&1; then printf %s magick; else printf %s convert; fi; }" \
  "__ensure_outdir() { mkdir -p \"\$(dirname -- \"\$1\")\"; }" \
  "" \
  "# ---------- img2jpg: near-full quality JPEG ----------" \
  "img2jpg() {" \
  "  [ \"\$#\" -ge 1 ] || { echo 'Usage: img2jpg <input> [output.jpg]'; return 2; }" \
  "  local in=\"\$1\"; shift; [ -f \"\$in\" ] || { echo \"img2jpg: input not found: \$in\"; return 2; }" \
  "  local base=\"\${in%.*}\"; local out=\"\${1:-\${base}.jpg}\"" \
  "  __ensure_outdir \"\$out\"" \
  "  local IM=\"\$(__imgmagick)\"" \
  "  \"\$IM\" \"\$in\" -auto-orient -strip -interlace JPEG -sampling-factor 4:2:0 -quality 92 \"\$out\"" \
  "}" \
  "" \
  "# ---------- img2jpg-small: 1080px max width ----------" \
  "img2jpg-small() {" \
  "  [ \"\$#\" -ge 1 ] || { echo 'Usage: img2jpg-small <input> [output.jpg]'; return 2; }" \
  "  local in=\"\$1\"; shift; [ -f \"\$in\" ] || { echo \"img2jpg-small: input not found: \$in\"; return 2; }" \
  "  local base=\"\${in%.*}\"; local out=\"\${1:-\${base}-1080w.jpg}\"" \
  "  __ensure_outdir \"\$out\"" \
  "  local IM=\"\$(__imgmagick)\"" \
  "  \"\$IM\" \"\$in\" -auto-orient -filter Lanczos -resize 1080x\\> -strip -interlace JPEG -sampling-factor 4:2:0 -quality 90 \"\$out\"" \
  "}" \
  "" \
  "# ---------- img2png: quantized 'lossy' PNG ----------" \
  "img2png() {" \
  "  [ \"\$#\" -ge 1 ] || { echo 'Usage: img2png <input> [output.png]'; return 2; }" \
  "  local in=\"\$1\"; shift; [ -f \"\$in\" ] || { echo \"img2png: input not found: \$in\"; return 2; }" \
  "  local base=\"\${in%.*}\"; local out=\"\${1:-\${base}.png}\"" \
  "  __ensure_outdir \"\$out\"" \
  "  local IM=\"\$(__imgmagick)\"" \
  "  \"\$IM\" \"\$in\" -auto-orient -strip -colors 256 -define png:compression-level=9 \"\$out\"" \
  "}" \
  "" \
  >> /etc/bash.bashrc


# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
    dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
    tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Create dev user
RUN useradd -m developer && echo "developer ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER developer
WORKDIR /home/developer


RUN mkdir -p /home/developer/.ssh && chmod 700 /home/developer/.ssh