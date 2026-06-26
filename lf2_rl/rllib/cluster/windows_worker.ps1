<#
.SYNOPSIS
    Join a Windows LF2 machine to the DGX Ray head as a rollout worker.

.DESCRIPTION
    Each Windows machine runs exactly one focused LF2 window and therefore
    advertises a single `LF2_WINDOW` custom resource. RLlib's env-placement
    config (`--lf2-window-resource`) ensures only rollout EnvRunners (never the
    GPU learner) are scheduled onto these nodes.

    Before running:
      * Little Fighter 2 must be running, in VS Mode, with the human player
        slots selected (matching --player-ids used for training).
      * Keep the LF2 window focused / on top while training.

.PARAMETER HeadAddress
    The Ray head address, e.g. "10.0.0.5:6379".

.EXAMPLE
    .\windows_worker.ps1 -HeadAddress 10.0.0.5:6379
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$HeadAddress,

    [int]$NumLf2Windows = 1,

    [int]$NumCpus = 0
)

$ErrorActionPreference = "Stop"

$resources = '{\"LF2_WINDOW\": ' + $NumLf2Windows + '}'

$rayArgs = @(
    "start",
    "--address=$HeadAddress",
    "--resources=$resources",
    "--num-gpus=0"          # actors do CPU env stepping only; learner has the GPU
)

if ($NumCpus -gt 0) {
    $rayArgs += "--num-cpus=$NumCpus"
}

Write-Host "Joining Ray head $HeadAddress with LF2_WINDOW=$NumLf2Windows ..."
ray @rayArgs

Write-Host ""
Write-Host "Worker joined. Ensure the LF2 window stays focused during training."
