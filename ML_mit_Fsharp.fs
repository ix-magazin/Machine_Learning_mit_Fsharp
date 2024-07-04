// Code zum Praxisartikel ML mit F#

// Listing 1: Import der TorchSharp Module

open TorchSharp
open type TorchSharp.torch
open type TorchSharp.TensorExtensionMethods
open type TorchSharp.torch.distributions

// Listing 2: Deklaration der Klasse SimpleModel

type SimpleModel() as this =
    inherit nn.Module<Tensor,Tensor>("SimpleModel")

    let lin1 = nn.Linear(750L, 100L)

    do
        this.RegisterComponents()

    override _.forward(input) = lin1.forward(input);


// Zufallszahlen generieren

let input = rand(750);
let model = SimpleModel();
model.forward(input);


// Listing 3: Implementieren des zweiten Layers und der ReLU-Funktion

type SimpleModel() as this =
    inherit nn.Module<Tensor,Tensor>("SimpleModel")

    let lin1 = nn.Linear(750L, 100L)
    let lin2 = nn.Linear(100L, 10L)

    do
        this.RegisterComponents()

    override _.forward(input) =

        use x = lin1.forward(input)
        use y = nn.functional.relu(x)
        lin2.forward(y)

let input = rand(750);
let model = SimpleModel();
model.forward(input);

let dataBatch = rand(32,1000)
let resultBatch = rand(32,10)
dataBatch.ToString()

let loss x y = nn.functional.mse_loss(x,y)
let pred = model.forward(dataBatch)
(loss pred resultBatch).item<single>()


// Listing 4: Implementieren des Lernprozesses

let learning_rate = 0.001f.ToScalar()

let pred = model.forward(dataBatch)
let output = loss pred resultBatch

model.zero_grad()
output.backward()

using(torch.no_grad()) (fun _ ->
    for param in model.parameters() do
        let grad = param.grad()
        match grad with
        | null -> ()
        | _ ->
            let update = grad.mul(learning_rate)
            param.sub_(update) |> ignore
)
(loss pred resultBatch).item<single>()
