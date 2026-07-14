<#
.SYNOPSIS
    Join a Windows LF2 machine to the DGX Ray head as a rollout worker.

.DESCRIPTION
    Each Windows machine runs exactly one focused LF2 window and therefore
    advertises a single `LF2_WINDOW` custom resource. RLlib's env-placement
    config (on by default in train.py) ensures only rollout EnvRunners are
    scheduled here while the gradient-computing learner stays on the DGX.

    Inference (policy forward pass) runs on the Windows side. If the
    Windows machine has a GPU, pass `-NumGpus 1` (or a fractional share)
    so Ray advertises it and RLlib's `--num-gpus-per-env-runner` can land
    inference on the GPU. No gradient is computed here; training stays
    on the DGX.

    Ray's multi-node mode is experimental on Windows and gated behind
    the `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` env var. This script sets
    it for the spawned `ray start` process automatically; the DGX head
    must also be started with the same variable in its environment
    (see dgx_head.sh).

    If the head and worker talk over a private overlay (Tailscale,
    WireGuard, ZeroTier), pass `-NodeIpAddress` with this machine's
    overlay IP so Ray binds and advertises the right interface;
    otherwise `Test-NetConnection` will succeed but Ray's handshake
    fails with a plain `ConnectionError`.

    Before running:
      * Little Fighter 2 must be running, in VS Mode, with the human player
        slots selected (matching --player-ids used for training).
      * Keep the LF2 window focused / on top while training.

.PARAMETER HeadAddress
    The Ray head address, e.g. "100.79.61.7:6379".

.PARAMETER NumLf2Windows
    How many LF2 game windows this machine hosts (= LF2_WINDOW resource
    count). Defaults to 1.

.PARAMETER NumGpus
    Number of GPUs (or fractional share) to advertise to Ray for policy
    inference. Defaults to 0 (CPU inference). Pass 1 if this box has a
    dedicated GPU; pass e.g. 0.25 to share one GPU across several
    co-located workers.

.PARAMETER NumCpus
    Override Ray's auto-detected CPU count (0 = let Ray decide).

.PARAMETER NodeIpAddress
    Optional. This machine's overlay IP that the head can reach. If
    omitted, Ray auto-detects, which usually picks the primary LAN NIC
    and can leave workers unreachable when the cluster runs over
    Tailscale / WireGuard.

.EXAMPLE
    .\windows_worker.ps1 -HeadAddress 100.79.61.7:6379 -NodeIpAddress 100.65.106.83 -NumGpus 1
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$HeadAddress,

    [int]$NumLf2Windows = 1,

    [double]$NumGpus = 0,

    [int]$NumCpus = 0,

    [string]$NodeIpAddress = ""
)

$ErrorActionPreference = "Stop"

# Ray gates multi-node clusters on Windows behind this env var. The DGX
# head must have the same variable set in its environment too.
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER = "1"

# Build the JSON literal with *real* double quotes (backtick-escaped inside
# a double-quoted PS string). PowerShell 5.x's native-EXE argument passing
# only wraps args in an outer ``"..."`` block when the arg contains a real
# ``"`` — so ``'{\"LF2_WINDOW\": 1}'`` (backslash-quote literals in a
# single-quoted string) gets passed to ``ray.exe`` verbatim and Ray sees
# ``{\"LF2_WINDOW\": 1}`` — invalid JSON. With backtick-escape the string
# actually contains ``{"LF2_WINDOW": 1}`` and PS then correctly escapes for
# native argv.
$resources = "{`"LF2_WINDOW`": $NumLf2Windows}"

$rayArgs = @(
    "start",
    "--address=$HeadAddress",
    "--resources=$resources",
    "--num-gpus=$NumGpus"
)

if ($NumCpus -gt 0) {
    $rayArgs += "--num-cpus=$NumCpus"
}

if ($NodeIpAddress -ne "") {
    $rayArgs += "--node-ip-address=$NodeIpAddress"
    Write-Host "Binding worker to $NodeIpAddress (from -NodeIpAddress)."
} else {
    Write-Host "Note: -NodeIpAddress not set. Ray will auto-detect the interface."
    Write-Host "      Pass this machine's Tailscale / WireGuard IP if the head is"
    Write-Host "      only reachable over that overlay (avoids silent Ray handshake"
    Write-Host "      failures despite TcpTestSucceeded=True on the head port)."
}

Write-Host "Joining Ray head $HeadAddress with LF2_WINDOW=$NumLf2Windows, GPUs=$NumGpus ..."
& ray @rayArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "ray start failed with exit code $LASTEXITCODE. Worker did NOT join the cluster."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Worker joined. Ensure the LF2 window stays focused during training."
Write-Host "Verify from the DGX head:  ray status   (look for 'LF2_WINDOW' in Total Usage)."
if ($NumGpus -le 0) {
    Write-Host "Tip: pass -NumGpus 1 to enable GPU policy inference (training still happens on the DGX)."
}
